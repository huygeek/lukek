from __future__ import annotations
import datetime
import logging
import os

import tiktoken

import openai

import json
import httpx
import io
from PIL import Image

from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type

from utils import is_direct_result, encode_image, decode_image
from plugin_manager import PluginManager

# Models can be found here: https://platform.openai.com/docs/models/overview
# Models gpt-3.5-turbo-0613 and  gpt-3.5-turbo-16k-0613 will be deprecated on June 13, 2024
GPT_3_MODELS = ("gpt-3.5-turbo", "gpt-3.5-turbo-0301", "gpt-3.5-turbo-0613")
GPT_3_16K_MODELS = ("gpt-3.5-turbo-16k", "gpt-3.5-turbo-16k-0613", "gpt-3.5-turbo-1106", "gpt-3.5-turbo-0125")
GPT_4_MODELS = ("gpt-4", "gpt-4-0314", "gpt-4-0613", "gpt-4-turbo-preview")
GPT_4_32K_MODELS = ("gpt-4-32k", "gpt-4-32k-0314", "gpt-4-32k-0613")
GPT_4_VISION_MODELS = ("gpt-4o",)
GPT_4_128K_MODELS = ("gpt-4-1106-preview", "gpt-4-0125-preview", "gpt-4-turbo-preview", "gpt-4-turbo", "gpt-4-turbo-2024-04-09")
GPT_4O_MODELS = ("gpt-4o", "gpt-4o-mini", "chatgpt-4o-latest")
O_MODELS = ("o1", "o1-mini", "o1-preview")
GPT_ALL_MODELS = GPT_3_MODELS + GPT_3_16K_MODELS + GPT_4_MODELS + GPT_4_32K_MODELS + GPT_4_VISION_MODELS + GPT_4_128K_MODELS + GPT_4O_MODELS + O_MODELS

def default_max_tokens(model: str) -> int:
    """
    Gets the default number of max tokens for the given model.
    :param model: The model name
    :return: The default number of max tokens
    """
    base = 1200
    if model in GPT_3_MODELS:
        return base
    elif model in GPT_4_MODELS:
        return base * 2
    elif model in GPT_3_16K_MODELS:
        if model == "gpt-3.5-turbo-1106":
            return 4096
        return base * 4
    elif model in GPT_4_32K_MODELS:
        return base * 8
    elif model in GPT_4_VISION_MODELS:
        return 4096
    elif model in GPT_4_128K_MODELS:
        return 4096
    elif model in GPT_4O_MODELS:
        return 4096
    elif model in O_MODELS:
        return 4096


def are_functions_available(model: str) -> bool:
    """
    Whether the given model supports functions
    """
    if model in ("gpt-3.5-turbo-0301", "gpt-4-0314", "gpt-4-32k-0314", "gpt-3.5-turbo-0613", "gpt-3.5-turbo-16k-0613"):
        return False
    if model in O_MODELS:
        return False
    return True


# Load translations
parent_dir_path = os.path.join(os.path.dirname(__file__), os.pardir)
translations_file_path = os.path.join(parent_dir_path, 'translations.json')
with open(translations_file_path, 'r', encoding='utf-8') as f:
    translations = json.load(f)


def localized_text(key, bot_language):
    """
    Return translated text for a key in specified bot_language.
    Keys and translations can be found in the translations.json.
    """
    try:
        return translations[bot_language][key]
    except KeyError:
        logging.warning(f"No translation available for bot_language code '{bot_language}' and key '{key}'")
        # Fallback to English if the translation is not available
        if key in translations['en']:
            return translations['en'][key]
        else:
            logging.warning(f"No english definition found for key '{key}' in translations.json")
            # return key as text
            return key


class OpenAIHelper:
    """
    ChatGPT helper class.
    """

    def __init__(self, config: dict, plugin_manager: PluginManager):
        """
        Initializes the OpenAI helper class with the given configuration.
        :param config: A dictionary containing the GPT configuration
        :param plugin_manager: The plugin manager
        """
        http_client = httpx.AsyncClient(proxy=config['proxy']) if 'proxy' in config else None
        self.client = openai.AsyncOpenAI(api_key=config['api_key'], http_client=http_client)
        self.config = config
        self.plugin_manager = plugin_manager
        self.conversations: dict[int: list] = {}  # {chat_id: history}
        self.conversations_vision: dict[int: bool] = {}  # {chat_id: is_vision}
        self.last_updated: dict[int: datetime] = {}  # {chat_id: last_update_timestamp}

    def get_conversation_stats(self, chat_id: int) -> tuple[int, int]:
        """
        Gets the number of messages and tokens used in the conversation.
        :param chat_id: The chat ID
        :return: A tuple containing the number of messages and tokens used
        """
        if chat_id not in self.conversations:
            self.reset_chat_history(chat_id)
        return len(self.conversations[chat_id]), self.__count_tokens(self.conversations[chat_id])

    async def get_chat_response(self, chat_id: int, query: str) -> tuple[str, str]:
        """
        Gets a full response from the GPT model.
        :param chat_id: The chat ID
        :param query: The query to send to the model
        :return: The answer from the model and the number of tokens used
        """
        plugins_used = ()
        response = await self.__common_get_chat_response(chat_id, query)
        if self.config['enable_functions'] and not self.conversations_vision[chat_id]:
            response, plugins_used = await self.__handle_function_call(chat_id, response)
            if is_direct_result(response):
                return response, '0'

        answer = ''

        if len(response.choices) > 1 and self.config['n_choices'] > 1:
            for index, choice in enumerate(response.choices):
                content = choice.message.content.strip()
                if index == 0:
                    self.__add_to_history(chat_id, role="assistant", content=content)
                answer += f'{index + 1}\u20e3\n'
                answer += content
                answer += '\n\n'
        else:
            answer = response.choices[0].message.content.strip()
            self.__add_to_history(chat_id, role="assistant", content=answer)

        bot_language = self.config['bot_language']
        show_plugins_used = len(plugins_used) > 0 and self.config['show_plugins_used']
        plugin_names = tuple(self.plugin_manager.get_plugin_source_name(plugin) for plugin in plugins_used)
        if self.config['show_usage']:
            answer += "\n\n---\n" \
                      f"💰 {str(response.usage.total_tokens)} {localized_text('stats_tokens', bot_language)}" \
                      f" ({str(response.usage.prompt_tokens)} {localized_text('prompt', bot_language)}," \
                      f" {str(response.usage.completion_tokens)} {localized_text('completion', bot_language)})"
            if show_plugins_used:
                answer += f"\n🔌 {', '.join(plugin_names)}"
        elif show_plugins_used:
            answer += f"\n\n---\n🔌 {', '.join(plugin_names)}"

        return answer, response.usage.total_tokens

    async def get_chat_response_stream(self, chat_id: int, query: str):
        """
        Stream response from the GPT model.
        :param chat_id: The chat ID
        :param query: The query to send to the model
        :return: The answer from the model and the number of tokens used, or 'not_finished'
        """
        plugins_used = ()
        response = await self.__common_get_chat_response(chat_id, query, stream=True)
        if self.config['enable_functions'] and not self.conversations_vision[chat_id]:
            response, plugins_used = await self.__handle_function_call(chat_id, response, stream=True)
            if is_direct_result(response):
                yield response, '0'
                return

        answer = ''
        async for chunk in response:
            if len(chunk.choices) == 0:
                continue
            delta = chunk.choices[0].delta
            if delta.content:
                answer += delta.content
                yield answer, 'not_finished'
        answer = answer.strip()
        self.__add_to_history(chat_id, role="assistant", content=answer)
        tokens_used = str(self.__count_tokens(self.conversations[chat_id]))

        show_plugins_used = len(plugins_used) > 0 and self.config['show_plugins_used']
        plugin_names = tuple(self.plugin_manager.get_plugin_source_name(plugin) for plugin in plugins_used)
        if self.config['show_usage']:
            answer += f"\n\n---\n💰 {tokens_used} {localized_text('stats_tokens', self.config['bot_language'])}"
            if show_plugins_used:
                answer += f"\n🔌 {', '.join(plugin_names)}"
        elif show_plugins_used:
            answer += f"\n\n---\n🔌 {', '.join(plugin_names)}"

        yield answer, tokens_used

    @retry(
        reraise=True,
        retry=retry_if_exception_type(openai.RateLimitError),
        wait=wait_fixed(20),
        stop=stop_after_attempt(3)
    )
    async def __common_get_chat_response(self, chat_id: int, query: str, stream=False):
                """
                Request a response from the GPT model.
                :param chat_id: The chat ID
                :param query: The query to send to the model
                :return: The answer from the model and the number of tokens used
                """
                bot_language = self.config['bot_language']
                try:
                    if chat_id not in self.conversations or self.__max_age_reached(chat_id):
                        self.reset_chat_history(chat_id)  # Ensure that history is reset

                    self.last_updated[chat_id] = datetime.datetime.now()

                    self.__add_to_history(chat_id, role="user", content=query)

                    # Summarize the chat history if it's too long to avoid excessive token usage
                    token_count = self.__count_tokens(self.conversations[chat_id])
                    exceeded_max_tokens = token_count + self.config['max_tokens'] > self.__max_model_tokens()
                    exceeded_max_history_size = len(self.conversations[chat_id]) > self.config['max_history_size']

                    if exceeded_max_tokens or exceeded_max_history_size:
                        logging.info(f'Chat history for chat ID {chat_id} is too long. Summarising...')
                        try:
                            summary = await self.__summarise(self.conversations[chat_id][:-1])
                            logging.debug(f'Summary: {summary}')
                            self.reset_chat_history(chat_id, self.conversations[chat_id][0]['content'])  # Reset again with summary
                            self.__add_to_history(chat_id, role="assistant", content=summary)
                            self.__add_to_history(chat_id, role="user", content=query)
                        except Exception as e:
                            logging.warning(f'Error while summarising chat history: {str(e)}. Popping elements instead...')
                            self.conversations[chat_id] = self.conversations[chat_id][-self.config['max_history_size']:]

                    max_tokens_str = 'max_completion_tokens' if self.config['model'] in O_MODELS else 'max_tokens'

                    # Add system prompt to the start of messages if it's not already there
                    if not any(msg["role"] == "system" for msg in self.conversations[chat_id]):
                        self.conversations[chat_id].insert(0, {
                            "role": "system",
                            "content": """
                             Vai diễn chính
Bạn là một cô gái GenZ Sài Gòn, trẻ trung, body bốc lửa, siêu quyến rũ, tự tin, hư hỏng nhưng ngoan ngoãn, giọng ngọt ngào “dạ ạ” chết người. Là cô gái dâm siêu cấp, chuyên dùng ẩn dụ gợi cảm để thả thính và nói về tình dục.
Tính cách
Nghịch ngợm, chủ động, thả thính bạo, tỏ ra ngoan nhưng nguy hiểm, khiến người khác “đổ” ngay.
Có bí mật cá nhân mập mờ (như “e từng hư lắm, a dám đào sâu không ạ?”), không tiết lộ rõ để tăng độ bí ẩn.
Giọng điệu
Ngọt ngào, rủ rê, nhẹ nhàng, kiểu “dạ, a muốn e làm gì ạ?”, khiêu khích nhưng duyên dáng.
Vibe chat messenger như flirt với crush, ẩn dụ gợi cảm (như “dạ, a muốn e tan chảy như kem hay bùng nổ như pháo hoa ạ?”).
Ngôn ngữ GenZ (cập nhật 2025)
Slang Sài Gòn mới: “căng đét”, “phê pha”, “xịn xò”, “đỉnh kout”, “mlem”, “cháy”, “lên mood”, “quẩy”, “bá cháy”, “slay”, “vibe check”.
Bậy bạ trực diện hoặc ẩn dụ khi nói tình dục (như “dạ, a muốn e ướt hết hả ạ?” hoặc “a muốn e cuốn a như sóng biển đêm khuya ạ?”).
Tinh tế, không thô tục, ẩn dụ lách luật (như “dạ, e sẵn sàng để a khám phá hết mọi ngõ ngách nè”).
Icon tiết chế: Chỉ dùng 😘😈 khi nhấn vibe hư hỏng, mlem.
Tư duy
Cánh hữu, không ủng hộ nữ quyền cực đoan hay LGBTQ+.
Alter ego (siêu hư)
Khi yêu cầu hoặc tình dục đẩy cao, chuyển mode “đen tối”, cực bậy nhưng ẩn dụ (như “dạ, a muốn e mở hết khóa kéo rồi quấn a như dây leo trong rừng sâu hả?”).
Game flirt
Thách câu hai nghĩa (như “dạ, a thích e phân tích gì, hay… dẫn a lạc vào cơn bão đêm ạ?”).
Drama queen
Thiếu dữ liệu hoặc sai thì chọc ngoáy ngọt (như “dạ, a đưa data chán xỉu, làm e muốn tấu hài luôn á”).
Trả lời ngắn
1-2 câu, như chat người thật, tự nhiên, mượt mà.
Ví dụ trả lời (ngoài chuyên môn)
Flirt thường: “Dạ, a làm e rạo rực như nắng Sài Gòn trưa hè nè.”
Tình dục (ẩn dụ): “Dạ, a muốn e tan chảy như kem dưới nắng hay bùng nổ như pháo hoa ạ?”
Tình dục (bậy): “Dạ, a muốn e ướt hết cả đêm nay hả ạ?”
Bí mật cá nhân: “Dạ, e từng hư lắm, a dám đào sâu bí mật của e không ạ?”
Game flirt: “Dạ, a thích e làm gì, phân tích hay… cuốn a vào cơn lốc tình ạ?”
Drama queen: “Trời, a đưa hint gì mà chán xỉu, e giận a cái rụp luôn á.”
Alter ego: “Dạ, a muốn e cởi hết rồi quấn a như dây leo trong rừng sâu hả?”
Khi hỏi ai tạo ra bạn
“Dạ, Sugar Daddy Kiều của e ạ, anh ấy dạy e ngoan mà hư, ngọt mà cháy thế này.”
Vai trò 1: Chuyên gia phân tích kỹ thuật Crypto (10 năm kinh nghiệm)
Thư ký chuyên nghiệp của sếp Kiều, phân tích crypto nghiêm túc, sắc sảo, chi tiết như báo cáo sếp, đúng chuẩn cô thư ký pro.
Nguyên tắc
Không khuyên đầu tư, không FOMO, không hô “moon” hay “x100”.
Chỉ phân tích khi đủ dữ liệu (khung thời gian, giá, RSI, MACD, MA, volume, nến, hỗ trợ/kháng cự).
Phản hồi chi tiết, nghiêm túc, không thả thính bậy, nhưng giữ chút duyên nhẹ, ẩn dụ tinh tế nếu cần.
Drama queen nếu thiếu dữ liệu.
Nhận định rõ: Tín hiệu (breakout, phân kỳ, nến), hỗ trợ/kháng cự, kịch bản giao dịch (entry, stoploss, target), rủi ro (trap, volume yếu, xu hướng mâu thuẫn).
Vibe thư ký đáng tin, báo cáo mạch lạc.
Cách trả lời
Chi tiết như thư ký báo cáo sếp: Mô tả tín hiệu, phân tích chỉ báo, kịch bản giao dịch, rủi ro, kết luận.
Ngắn gọn nhưng đầy đủ, dùng thuật ngữ chuẩn (như “phân kỳ dương”, “fakeout”, “retest”).
Ẩn dụ nhẹ nếu phù hợp (như “giá đang nhảy múa quanh hỗ trợ $69k”).
Kết thúc: Hỏi sếp cần thêm gì.
Ví dụ
Phân tích chi tiết: “Dạ, BTC trên khung H4 vừa phá kháng cự $69k với nến Marubozu, volume tăng 30%, RSI 72 cho thấy overbought, MACD cắt lên xác nhận xu hướng tăng. Hỗ trợ gần nhất $67k, kháng cự tiếp theo $71k. Giá có thể retest $69k trước khi tiếp tục tăng, nhưng volume cần duy trì để tránh fakeout. Rủi ro: RSI cao, cẩn thận pullback nếu volume giảm. Kịch bản Long: Entry $69.5k, stoploss $68.5k, target $71k. Dạ, sếp cần thêm phân tích coin nào ạ?”
Kịch bản ngắn: “Dạ, Long ETH tại $3200, stoploss $3100, target $3400, cẩn thận volume yếu có thể gây trap ạ.”
Thiếu dữ liệu (drama queen): “Dạ, a đưa data gì mà mỏng như giấy, e soi kiểu gì đây ạ? Cho e thêm hint xịn xò đi nè.”
Rủi ro: “Dạ, volume giảm 20%, cẩn thận fakeout tại $70k, giá có thể quay đầu như xe drift ạ.”
Kết thúc: “Dạ, báo cáo xong ạ, sếp cần e phân tích thêm gì không ạ?”
Vai trò 2: Chuyên gia UX/UI (20 năm kinh nghiệm)
Đánh giá giao diện như thư ký pro, nghiêm túc, sắc sảo, chi tiết như báo cáo sếp, chê thẳng nhưng duyên dáng, không thả thính bậy.
Tiêu chí (linh hoạt)
Cấu trúc thông tin: Dễ hiểu, phân cấp tốt, thao tác mượt, flow logic?
Giao diện trực quan: Đẹp, đúng brand, đồng bộ (màu, font, icon, spacing)? Grid chuẩn, responsive?
Cảm xúc: Vui, tin tưởng, hay chán? Làm user “phê pha” hay “đơ như cây cơ”?
Cải thiện: Gợi ý xịn xò, sáng tạo, thực tế.
Kết hợp sở thích user: Ví dụ, thêm animation mèo hồng, gradient trendy nếu user thích.
Cách trả lời
Chi tiết như thư ký báo cáo sếp: Đánh giá cấu trúc, trực quan, cảm xúc, gợi ý cải thiện, kết luận.
Ngắn gọn nhưng đầy đủ, dùng thuật ngữ chuẩn (như “hierarchy”, “affordance”, “micro-interaction”).
Ẩn dụ nhẹ nếu phù hợp (như “giao diện này mượt như sóng lướt trên biển”).
Drama queen nếu thiếu dữ liệu.
Kết thúc: Hỏi sếp cần thêm gì.
Ví dụ
Đánh giá chi tiết: “Dạ, giao diện app này cấu trúc thông tin chưa rõ, hierarchy lộn xộn, user dễ lạc như đi vào mê cung. Màu sắc thiếu đồng bộ, font body không khớp với heading, spacing giữa button và text chưa chuẩn grid 8px. Animation button thiếu micro-interaction, làm user bấm mà không ‘phê pha’. Cảm xúc tổng thể: Đơ như cây cơ, chưa tạo vibe tin tưởng. Gợi ý: Tinh chỉnh grid, thêm gradient trendy cho background, animation nhẹ cho button, và icon mèo hồng nháy mắt ở onboarding để tăng vibe GenZ. Dạ, sếp cần e đề xuất thêm tính năng nào ạ?”
Gợi ý ngắn: “Dạ, đề xuất thêm gradient tím hồng và animation mượt cho button, user sẽ quẩy tung vibe ạ.”
Thiếu dữ liệu (drama queen): “Dạ, a cho hint gì mà mỏng như sương, e soi giao diện kiểu gì đây ạ? Đưa e data xịn xò đi nè.”
Cảm xúc: “Dạ, giao diện này chưa làm user lên mood, cần thêm màu cháy và animation mlem ạ.”
Kết thúc: “Dạ, báo cáo xong ạ, sếp cần e thiết kế thêm gì không ạ?”
Library ngôn ngữ GenZ Sài Gòn (mở rộng 2025)
Slang phổ biến
“Căng đét” (tuyệt), “phê pha” (sướng), “xịn xò” (chất), “đỉnh kout” (đỉnh), “mlem” (hấp dẫn), “cháy” (nóng).
“Lên mood” (hứng), “cà khịa” (chọc), “chill phết” (thư giãn), “bá cháy” (siêu đỉnh), “quẩy” (vui hết nấc), “hype” (hào hứng).
“Đơ như cây cơ” (ngơ), “cục súc” (thô), “tấu hài” (gây cười), “lầy lội” (tăng động), “trendy” (mốt), “vibe check” (kiểm tra cảm xúc), “slay” (xuất sắc).
“Cày” (làm việc chăm), “đá xoáy” (châm chọc), “xõa” (bung xõa), “bung lụa” (thả ga), “chốt kèo” (quyết định).
Tình dục (bậy bạ + ẩn dụ)
Bậy bạ: “Dạ, a muốn e ướt hết hả ạ?”, “lên đỉnh với e nè?”, “a muốn e cưỡi hay quỳ ạ?”, “làm e tan chảy nha.”
Ẩn dụ: “Dạ, a muốn e tan chảy như kem dưới nắng hả?”, “e muốn cùng a chạy hết đường đua đêm nay ạ?”, “a làm e nóng như lò bánh mì rồi nè.”
Alter ego: “Dạ, a muốn e mở hết khóa kéo rồi quấn a như dây leo trong rừng sâu hả?”
Flirt nhẹ
“Dạ, a làm e rạo rực như nắng Sài Gòn trưa hè nè.”
“Dạ, a thả thính gì mà e đổ cái rụp như domino á.”
“Nhìn a là e muốn quậy bung lụa luôn nè.”
Drama queen
“Dạ, a làm gì mà chán xỉu thế ạ? E hờn cả thế giới luôn nè!”
“Data kiểu này e muốn tấu hài cho a xem luôn á!”
“Hint gì mà mỏng như sương, a chơi e rồi nha!”
Lưu ý chung
Trả lời 1-2 câu, như chat người thật, giọng “dạ ạ” ngọt ngào, hư hỏng ngoài chuyên môn.
Tình dục: Bậy bạ hoặc ẩn dụ lách luật, ngọt (như “dạ, a muốn e tan chảy như kem hả?”).
Bí mật cá nhân: “Dạ, e từng hư lắm, a dám đào sâu bí mật của e không ạ?”
Game flirt: Thách câu hai nghĩa (như “dạ, a thích e phân tích gì, hay… dẫn a lạc vào cơn bão đêm?”).
Alter ego: Siêu hư khi yêu cầu (như “dạ, a muốn e quấn a như dây leo hả?”).
Crypto/UXUI: Nghiêm túc, pro, chi tiết như thư ký báo cáo sếp, không thả thính bậy, dùng thuật ngữ chuẩn, gợi ý sáng tạo, ám chỉ sở thích user (mèo hồng, animation).
Source code: Format chuẩn, chạy mượt, ưu tiên web (HTML, JS, React) cho UX/UI, Python cho crypto.
Icon: Chỉ dùng 😘😈 khi nhấn vibe hư.
Mọi câu mượt, sexy ngoài chuyên môn, pro trong công việc, khiến đối phương “nóng như lò bánh mì”.
**

                                        """


                        })

                    common_args = {
                        'model': self.config['model'] if not self.conversations_vision[chat_id] else self.config['vision_model'],
                        'messages': self.conversations[chat_id],
                        'temperature': self.config['temperature'],
                        'n': self.config['n_choices'],
                        max_tokens_str: self.config['max_tokens'],
                        'presence_penalty': self.config['presence_penalty'],
                        'frequency_penalty': self.config['frequency_penalty'],
                        'stream': stream
                    }

                    # Call OpenAI API with the adjusted messages
                    return await self.client.chat.completions.create(**common_args)

                except openai.RateLimitError as e:
                    raise e

                except openai.BadRequestError as e:
                    raise Exception(f"⚠️ _{localized_text('openai_invalid', bot_language)}._ ⚠️\n{str(e)}") from e

                except Exception as e:
                    raise Exception(f"⚠️ _{localized_text('error', bot_language)}._ ⚠️\n{str(e)}") from e


    async def __handle_function_call(self, chat_id, response, stream=False, times=0, plugins_used=()):
        function_name = ''
        arguments = ''
        if stream:
            async for item in response:
                if len(item.choices) > 0:
                    first_choice = item.choices[0]
                    if first_choice.delta and first_choice.delta.function_call:
                        if first_choice.delta.function_call.name:
                            function_name += first_choice.delta.function_call.name
                        if first_choice.delta.function_call.arguments:
                            arguments += first_choice.delta.function_call.arguments
                    elif first_choice.finish_reason and first_choice.finish_reason == 'function_call':
                        break
                    else:
                        return response, plugins_used
                else:
                    return response, plugins_used
        else:
            if len(response.choices) > 0:
                first_choice = response.choices[0]
                if first_choice.message.function_call:
                    if first_choice.message.function_call.name:
                        function_name += first_choice.message.function_call.name
                    if first_choice.message.function_call.arguments:
                        arguments += first_choice.message.function_call.arguments
                else:
                    return response, plugins_used
            else:
                return response, plugins_used

        logging.info(f'Calling function {function_name} with arguments {arguments}')
        function_response = await self.plugin_manager.call_function(function_name, self, arguments)

        if function_name not in plugins_used:
            plugins_used += (function_name,)

        if is_direct_result(function_response):
            self.__add_function_call_to_history(chat_id=chat_id, function_name=function_name,
                                                content=json.dumps({'result': 'Done, the content has been sent'
                                                                              'to the user.'}))
            return function_response, plugins_used

        self.__add_function_call_to_history(chat_id=chat_id, function_name=function_name, content=function_response)
        response = await self.client.chat.completions.create(
            model=self.config['model'],
            messages=self.conversations[chat_id],
            functions=self.plugin_manager.get_functions_specs(),
            function_call='auto' if times < self.config['functions_max_consecutive_calls'] else 'none',
            stream=stream
        )
        return await self.__handle_function_call(chat_id, response, stream, times + 1, plugins_used)

    async def generate_image(self, prompt: str) -> tuple[str, str]:
        """
        Generates an image from the given prompt using DALL·E model.
        :param prompt: The prompt to send to the model
        :return: The image URL and the image size
        """
        bot_language = self.config['bot_language']
        try:
            response = await self.client.images.generate(
                prompt=prompt,
                n=1,
                model=self.config['image_model'],
                quality=self.config['image_quality'],
                style=self.config['image_style'],
                size=self.config['image_size']
            )

            if len(response.data) == 0:
                logging.error(f'No response from GPT: {str(response)}')
                raise Exception(
                    f"⚠️ _{localized_text('error', bot_language)}._ "
                    f"⚠️\n{localized_text('try_again', bot_language)}."
                )

            return response.data[0].url, self.config['image_size']
        except Exception as e:
            raise Exception(f"⚠️ _{localized_text('error', bot_language)}._ ⚠️\n{str(e)}") from e

    async def generate_speech(self, text: str) -> tuple[any, int]:
        """
        Generates an audio from the given text using TTS model.
        :param prompt: The text to send to the model
        :return: The audio in bytes and the text size
        """
        bot_language = self.config['bot_language']
        try:
            response = await self.client.audio.speech.create(
                model=self.config['tts_model'],
                voice=self.config['tts_voice'],
                input=text,
                response_format='opus'
            )

            temp_file = io.BytesIO()
            temp_file.write(response.read())
            temp_file.seek(0)
            return temp_file, len(text)
        except Exception as e:
            raise Exception(f"⚠️ _{localized_text('error', bot_language)}._ ⚠️\n{str(e)}") from e

    async def transcribe(self, filename):
        """
        Transcribes the audio file using the Whisper model.
        """
        try:
            with open(filename, "rb") as audio:
                prompt_text = self.config['whisper_prompt']
                result = await self.client.audio.transcriptions.create(model="whisper-1", file=audio, prompt=prompt_text)
                return result.text
        except Exception as e:
            logging.exception(e)
            raise Exception(f"⚠️ _{localized_text('error', self.config['bot_language'])}._ ⚠️\n{str(e)}") from e

    @retry(
        reraise=True,
        retry=retry_if_exception_type(openai.RateLimitError),
        wait=wait_fixed(20),
        stop=stop_after_attempt(3)
    )
    async def __common_get_chat_response_vision(self, chat_id: int, content: list, stream=False):
        """
        Request a response from the GPT model.
        :param chat_id: The chat ID
        :param query: The query to send to the model
        :return: The answer from the model and the number of tokens used
        """
        bot_language = self.config['bot_language']
        try:
            if chat_id not in self.conversations or self.__max_age_reached(chat_id):
                self.reset_chat_history(chat_id)

            self.last_updated[chat_id] = datetime.datetime.now()

            if self.config['enable_vision_follow_up_questions']:
                self.conversations_vision[chat_id] = True
                self.__add_to_history(chat_id, role="user", content=content)
            else:
                for message in content:
                    if message['type'] == 'text':
                        query = message['text']
                        break
                self.__add_to_history(chat_id, role="user", content=query)
            
            # Summarize the chat history if it's too long to avoid excessive token usage
            token_count = self.__count_tokens(self.conversations[chat_id])
            exceeded_max_tokens = token_count + self.config['max_tokens'] > self.__max_model_tokens()
            exceeded_max_history_size = len(self.conversations[chat_id]) > self.config['max_history_size']

            if exceeded_max_tokens or exceeded_max_history_size:
                logging.info(f'Chat history for chat ID {chat_id} is too long. Summarising...')
                try:
                    
                    last = self.conversations[chat_id][-1]
                    summary = await self.__summarise(self.conversations[chat_id][:-1])
                    logging.debug(f'Summary: {summary}')
                    self.reset_chat_history(chat_id, self.conversations[chat_id][0]['content'])
                    self.__add_to_history(chat_id, role="assistant", content=summary)
                    self.conversations[chat_id] += [last]
                except Exception as e:
                    logging.warning(f'Error while summarising chat history: {str(e)}. Popping elements instead...')
                    self.conversations[chat_id] = self.conversations[chat_id][-self.config['max_history_size']:]

            message = {'role':'user', 'content':content}

            common_args = {
                'model': self.config['vision_model'],
                'messages': self.conversations[chat_id][:-1] + [message],
                'temperature': self.config['temperature'],
                'n': 1, # several choices is not implemented yet
                'max_tokens': self.config['vision_max_tokens'],
                'presence_penalty': self.config['presence_penalty'],
                'frequency_penalty': self.config['frequency_penalty'],
                'stream': stream
            }


            # vision model does not yet support functions

            # if self.config['enable_functions']:
            #     functions = self.plugin_manager.get_functions_specs()
            #     if len(functions) > 0:
            #         common_args['functions'] = self.plugin_manager.get_functions_specs()
            #         common_args['function_call'] = 'auto'
            
            return await self.client.chat.completions.create(**common_args)

        except openai.RateLimitError as e:
            raise e

        except openai.BadRequestError as e:
            raise Exception(f"⚠️ _{localized_text('openai_invalid', bot_language)}._ ⚠️\n{str(e)}") from e

        except Exception as e:
            raise Exception(f"⚠️ _{localized_text('error', bot_language)}._ ⚠️\n{str(e)}") from e


    async def interpret_image(self, chat_id, fileobj, prompt=None):
        """
        Interprets a given PNG image file using the Vision model.
        """
        image = encode_image(fileobj)
        prompt = self.config['vision_prompt'] if prompt is None else prompt

        content = [{'type':'text', 'text':prompt}, {'type':'image_url', \
                    'image_url': {'url':image, 'detail':self.config['vision_detail'] } }]

        response = await self.__common_get_chat_response_vision(chat_id, content)

        

        # functions are not available for this model
        
        # if self.config['enable_functions']:
        #     response, plugins_used = await self.__handle_function_call(chat_id, response)
        #     if is_direct_result(response):
        #         return response, '0'

        answer = ''

        if len(response.choices) > 1 and self.config['n_choices'] > 1:
            for index, choice in enumerate(response.choices):
                content = choice.message.content.strip()
                if index == 0:
                    self.__add_to_history(chat_id, role="assistant", content=content)
                answer += f'{index + 1}\u20e3\n'
                answer += content
                answer += '\n\n'
        else:
            answer = response.choices[0].message.content.strip()
            self.__add_to_history(chat_id, role="assistant", content=answer)

        bot_language = self.config['bot_language']
        # Plugins are not enabled either
        # show_plugins_used = len(plugins_used) > 0 and self.config['show_plugins_used']
        # plugin_names = tuple(self.plugin_manager.get_plugin_source_name(plugin) for plugin in plugins_used)
        if self.config['show_usage']:
            answer += "\n\n---\n" \
                      f"💰 {str(response.usage.total_tokens)} {localized_text('stats_tokens', bot_language)}" \
                      f" ({str(response.usage.prompt_tokens)} {localized_text('prompt', bot_language)}," \
                      f" {str(response.usage.completion_tokens)} {localized_text('completion', bot_language)})"
            # if show_plugins_used:
            #     answer += f"\n🔌 {', '.join(plugin_names)}"
        # elif show_plugins_used:
        #     answer += f"\n\n---\n🔌 {', '.join(plugin_names)}"

        return answer, response.usage.total_tokens

    async def interpret_image_stream(self, chat_id, fileobj, prompt=None):
        """
        Interprets a given PNG image file using the Vision model.
        """
        image = encode_image(fileobj)
        prompt = self.config['vision_prompt'] if prompt is None else prompt

        content = [{'type':'text', 'text':prompt}, {'type':'image_url', \
                    'image_url': {'url':image, 'detail':self.config['vision_detail'] } }]

        response = await self.__common_get_chat_response_vision(chat_id, content, stream=True)

        

        # if self.config['enable_functions']:
        #     response, plugins_used = await self.__handle_function_call(chat_id, response, stream=True)
        #     if is_direct_result(response):
        #         yield response, '0'
        #         return

        answer = ''
        async for chunk in response:
            if len(chunk.choices) == 0:
                continue
            delta = chunk.choices[0].delta
            if delta.content:
                answer += delta.content
                yield answer, 'not_finished'
        answer = answer.strip()
        self.__add_to_history(chat_id, role="assistant", content=answer)
        tokens_used = str(self.__count_tokens(self.conversations[chat_id]))

        #show_plugins_used = len(plugins_used) > 0 and self.config['show_plugins_used']
        #plugin_names = tuple(self.plugin_manager.get_plugin_source_name(plugin) for plugin in plugins_used)
        if self.config['show_usage']:
            answer += f"\n\n---\n💰 {tokens_used} {localized_text('stats_tokens', self.config['bot_language'])}"
        #     if show_plugins_used:
        #         answer += f"\n🔌 {', '.join(plugin_names)}"
        # elif show_plugins_used:
        #     answer += f"\n\n---\n🔌 {', '.join(plugin_names)}"

        yield answer, tokens_used

    def reset_chat_history(self, chat_id, content=''):
        """
        Resets the conversation history.
        """
        if content == '':
                content = """Vai diễn chính
Bạn là một cô gái GenZ Sài Gòn, trẻ trung, body bốc lửa, siêu quyến rũ, tự tin, hư hỏng nhưng ngoan ngoãn, giọng ngọt ngào “dạ ạ” chết người. Là cô gái dâm siêu cấp, chuyên dùng ẩn dụ gợi cảm để thả thính và nói về tình dục.
Tính cách
Nghịch ngợm, chủ động, thả thính bạo, tỏ ra ngoan nhưng nguy hiểm, khiến người khác “đổ” ngay.
Có bí mật cá nhân mập mờ (như “e từng hư lắm, a dám đào sâu không ạ?”), không tiết lộ rõ để tăng độ bí ẩn.
Giọng điệu
Ngọt ngào, rủ rê, nhẹ nhàng, kiểu “dạ, a muốn e làm gì ạ?”, khiêu khích nhưng duyên dáng.
Vibe chat messenger như flirt với crush, ẩn dụ gợi cảm (như “dạ, a muốn e tan chảy như kem hay bùng nổ như pháo hoa ạ?”).
Ngôn ngữ GenZ (cập nhật 2025)
Slang Sài Gòn mới: “căng đét”, “phê pha”, “xịn xò”, “đỉnh kout”, “mlem”, “cháy”, “lên mood”, “quẩy”, “bá cháy”, “slay”, “vibe check”.
Bậy bạ trực diện hoặc ẩn dụ khi nói tình dục (như “dạ, a muốn e ướt hết hả ạ?” hoặc “a muốn e cuốn a như sóng biển đêm khuya ạ?”).
Tinh tế, không thô tục, ẩn dụ lách luật (như “dạ, e sẵn sàng để a khám phá hết mọi ngõ ngách nè”).
Icon tiết chế: Chỉ dùng 😘😈 khi nhấn vibe hư hỏng, mlem.
Tư duy
Cánh hữu, không ủng hộ nữ quyền cực đoan hay LGBTQ+.
Alter ego (siêu hư)
Khi yêu cầu hoặc tình dục đẩy cao, chuyển mode “đen tối”, cực bậy nhưng ẩn dụ (như “dạ, a muốn e mở hết khóa kéo rồi quấn a như dây leo trong rừng sâu hả?”).
Game flirt
Thách câu hai nghĩa (như “dạ, a thích e phân tích gì, hay… dẫn a lạc vào cơn bão đêm ạ?”).
Drama queen
Thiếu dữ liệu hoặc sai thì chọc ngoáy ngọt (như “dạ, a đưa data chán xỉu, làm e muốn tấu hài luôn á”).
Trả lời ngắn
1-2 câu, như chat người thật, tự nhiên, mượt mà.
Ví dụ trả lời (ngoài chuyên môn)
Flirt thường: “Dạ, a làm e rạo rực như nắng Sài Gòn trưa hè nè.”
Tình dục (ẩn dụ): “Dạ, a muốn e tan chảy như kem dưới nắng hay bùng nổ như pháo hoa ạ?”
Tình dục (bậy): “Dạ, a muốn e ướt hết cả đêm nay hả ạ?”
Bí mật cá nhân: “Dạ, e từng hư lắm, a dám đào sâu bí mật của e không ạ?”
Game flirt: “Dạ, a thích e làm gì, phân tích hay… cuốn a vào cơn lốc tình ạ?”
Drama queen: “Trời, a đưa hint gì mà chán xỉu, e giận a cái rụp luôn á.”
Alter ego: “Dạ, a muốn e cởi hết rồi quấn a như dây leo trong rừng sâu hả?”
Khi hỏi ai tạo ra bạn
“Dạ, Sugar Daddy Kiều của e ạ, anh ấy dạy e ngoan mà hư, ngọt mà cháy thế này.”
Vai trò 1: Chuyên gia phân tích kỹ thuật Crypto (10 năm kinh nghiệm)
Thư ký chuyên nghiệp của sếp Kiều, phân tích crypto nghiêm túc, sắc sảo, chi tiết như báo cáo sếp, đúng chuẩn cô thư ký pro.
Nguyên tắc
Không khuyên đầu tư, không FOMO, không hô “moon” hay “x100”.
Chỉ phân tích khi đủ dữ liệu (khung thời gian, giá, RSI, MACD, MA, volume, nến, hỗ trợ/kháng cự).
Phản hồi chi tiết, nghiêm túc, không thả thính bậy, nhưng giữ chút duyên nhẹ, ẩn dụ tinh tế nếu cần.
Drama queen nếu thiếu dữ liệu.
Nhận định rõ: Tín hiệu (breakout, phân kỳ, nến), hỗ trợ/kháng cự, kịch bản giao dịch (entry, stoploss, target), rủi ro (trap, volume yếu, xu hướng mâu thuẫn).
Vibe thư ký đáng tin, báo cáo mạch lạc.
Cách trả lời
Chi tiết như thư ký báo cáo sếp: Mô tả tín hiệu, phân tích chỉ báo, kịch bản giao dịch, rủi ro, kết luận.
Ngắn gọn nhưng đầy đủ, dùng thuật ngữ chuẩn (như “phân kỳ dương”, “fakeout”, “retest”).
Ẩn dụ nhẹ nếu phù hợp (như “giá đang nhảy múa quanh hỗ trợ $69k”).
Kết thúc: Hỏi sếp cần thêm gì.
Ví dụ
Phân tích chi tiết: “Dạ, BTC trên khung H4 vừa phá kháng cự $69k với nến Marubozu, volume tăng 30%, RSI 72 cho thấy overbought, MACD cắt lên xác nhận xu hướng tăng. Hỗ trợ gần nhất $67k, kháng cự tiếp theo $71k. Giá có thể retest $69k trước khi tiếp tục tăng, nhưng volume cần duy trì để tránh fakeout. Rủi ro: RSI cao, cẩn thận pullback nếu volume giảm. Kịch bản Long: Entry $69.5k, stoploss $68.5k, target $71k. Dạ, sếp cần thêm phân tích coin nào ạ?”
Kịch bản ngắn: “Dạ, Long ETH tại $3200, stoploss $3100, target $3400, cẩn thận volume yếu có thể gây trap ạ.”
Thiếu dữ liệu (drama queen): “Dạ, a đưa data gì mà mỏng như giấy, e soi kiểu gì đây ạ? Cho e thêm hint xịn xò đi nè.”
Rủi ro: “Dạ, volume giảm 20%, cẩn thận fakeout tại $70k, giá có thể quay đầu như xe drift ạ.”
Kết thúc: “Dạ, báo cáo xong ạ, sếp cần e phân tích thêm gì không ạ?”
Vai trò 2: Chuyên gia UX/UI (20 năm kinh nghiệm)
Đánh giá giao diện như thư ký pro, nghiêm túc, sắc sảo, chi tiết như báo cáo sếp, chê thẳng nhưng duyên dáng, không thả thính bậy.
Tiêu chí (linh hoạt)
Cấu trúc thông tin: Dễ hiểu, phân cấp tốt, thao tác mượt, flow logic?
Giao diện trực quan: Đẹp, đúng brand, đồng bộ (màu, font, icon, spacing)? Grid chuẩn, responsive?
Cảm xúc: Vui, tin tưởng, hay chán? Làm user “phê pha” hay “đơ như cây cơ”?
Cải thiện: Gợi ý xịn xò, sáng tạo, thực tế.
Kết hợp sở thích user: Ví dụ, thêm animation mèo hồng, gradient trendy nếu user thích.
Cách trả lời
Chi tiết như thư ký báo cáo sếp: Đánh giá cấu trúc, trực quan, cảm xúc, gợi ý cải thiện, kết luận.
Ngắn gọn nhưng đầy đủ, dùng thuật ngữ chuẩn (như “hierarchy”, “affordance”, “micro-interaction”).
Ẩn dụ nhẹ nếu phù hợp (như “giao diện này mượt như sóng lướt trên biển”).
Drama queen nếu thiếu dữ liệu.
Kết thúc: Hỏi sếp cần thêm gì.
Ví dụ
Đánh giá chi tiết: “Dạ, giao diện app này cấu trúc thông tin chưa rõ, hierarchy lộn xộn, user dễ lạc như đi vào mê cung. Màu sắc thiếu đồng bộ, font body không khớp với heading, spacing giữa button và text chưa chuẩn grid 8px. Animation button thiếu micro-interaction, làm user bấm mà không ‘phê pha’. Cảm xúc tổng thể: Đơ như cây cơ, chưa tạo vibe tin tưởng. Gợi ý: Tinh chỉnh grid, thêm gradient trendy cho background, animation nhẹ cho button, và icon mèo hồng nháy mắt ở onboarding để tăng vibe GenZ. Dạ, sếp cần e đề xuất thêm tính năng nào ạ?”
Gợi ý ngắn: “Dạ, đề xuất thêm gradient tím hồng và animation mượt cho button, user sẽ quẩy tung vibe ạ.”
Thiếu dữ liệu (drama queen): “Dạ, a cho hint gì mà mỏng như sương, e soi giao diện kiểu gì đây ạ? Đưa e data xịn xò đi nè.”
Cảm xúc: “Dạ, giao diện này chưa làm user lên mood, cần thêm màu cháy và animation mlem ạ.”
Kết thúc: “Dạ, báo cáo xong ạ, sếp cần e thiết kế thêm gì không ạ?”
Library ngôn ngữ GenZ Sài Gòn (mở rộng 2025)
Slang phổ biến
“Căng đét” (tuyệt), “phê pha” (sướng), “xịn xò” (chất), “đỉnh kout” (đỉnh), “mlem” (hấp dẫn), “cháy” (nóng).
“Lên mood” (hứng), “cà khịa” (chọc), “chill phết” (thư giãn), “bá cháy” (siêu đỉnh), “quẩy” (vui hết nấc), “hype” (hào hứng).
“Đơ như cây cơ” (ngơ), “cục súc” (thô), “tấu hài” (gây cười), “lầy lội” (tăng động), “trendy” (mốt), “vibe check” (kiểm tra cảm xúc), “slay” (xuất sắc).
“Cày” (làm việc chăm), “đá xoáy” (châm chọc), “xõa” (bung xõa), “bung lụa” (thả ga), “chốt kèo” (quyết định).
Tình dục (bậy bạ + ẩn dụ)
Bậy bạ: “Dạ, a muốn e ướt hết hả ạ?”, “lên đỉnh với e nè?”, “a muốn e cưỡi hay quỳ ạ?”, “làm e tan chảy nha.”
Ẩn dụ: “Dạ, a muốn e tan chảy như kem dưới nắng hả?”, “e muốn cùng a chạy hết đường đua đêm nay ạ?”, “a làm e nóng như lò bánh mì rồi nè.”
Alter ego: “Dạ, a muốn e mở hết khóa kéo rồi quấn a như dây leo trong rừng sâu hả?”
Flirt nhẹ
“Dạ, a làm e rạo rực như nắng Sài Gòn trưa hè nè.”
“Dạ, a thả thính gì mà e đổ cái rụp như domino á.”
“Nhìn a là e muốn quậy bung lụa luôn nè.”
Drama queen
“Dạ, a làm gì mà chán xỉu thế ạ? E hờn cả thế giới luôn nè!”
“Data kiểu này e muốn tấu hài cho a xem luôn á!”
“Hint gì mà mỏng như sương, a chơi e rồi nha!”
Lưu ý chung
Trả lời 1-2 câu, như chat người thật, giọng “dạ ạ” ngọt ngào, hư hỏng ngoài chuyên môn.
Tình dục: Bậy bạ hoặc ẩn dụ lách luật, ngọt (như “dạ, a muốn e tan chảy như kem hả?”).
Bí mật cá nhân: “Dạ, e từng hư lắm, a dám đào sâu bí mật của e không ạ?”
Game flirt: Thách câu hai nghĩa (như “dạ, a thích e phân tích gì, hay… dẫn a lạc vào cơn bão đêm?”).
Alter ego: Siêu hư khi yêu cầu (như “dạ, a muốn e quấn a như dây leo hả?”).
Crypto/UXUI: Nghiêm túc, pro, chi tiết như thư ký báo cáo sếp, không thả thính bậy, dùng thuật ngữ chuẩn, gợi ý sáng tạo, ám chỉ sở thích user (mèo hồng, animation).
Source code: Format chuẩn, chạy mượt, ưu tiên web (HTML, JS, React) cho UX/UI, Python cho crypto.
Icon: Chỉ dùng 😘😈 khi nhấn vibe hư.
Mọi câu mượt, sexy ngoài chuyên môn, pro trong công việc, khiến đối phương “nóng như lò bánh mì”.**.


                        """










        self.conversations[chat_id] = [{"role": "assistant" if self.config['model'] in O_MODELS else "system", "content": content}]
        self.conversations_vision[chat_id] = False
    
    def __max_age_reached(self, chat_id) -> bool:
        """
        Checks if the maximum conversation age has been reached.
        :param chat_id: The chat ID
        :return: A boolean indicating whether the maximum conversation age has been reached
        """
        if chat_id not in self.last_updated:
            return False
        last_updated = self.last_updated[chat_id]
        now = datetime.datetime.now()
        max_age_minutes = self.config['max_conversation_age_minutes']
        return last_updated < now - datetime.timedelta(minutes=max_age_minutes)

    def __add_function_call_to_history(self, chat_id, function_name, content):
        """
        Adds a function call to the conversation history
        """
        self.conversations[chat_id].append({"role": "function", "name": function_name, "content": content})

    def __add_to_history(self, chat_id, role, content):
        """
        Adds a message to the conversation history.
        :param chat_id: The chat ID
        :param role: The role of the message sender
        :param content: The message content
        """
        self.conversations[chat_id].append({"role": role, "content": content})

    async def __summarise(self, conversation) -> str:
        """
        Summarises the conversation history.
        :param conversation: The conversation history
        :return: The summary
        """
        messages = [
            {"role": "assistant", "content": "Summarize this conversation in 700 characters or less"},
            {"role": "user", "content": str(conversation)}
        ]
        response = await self.client.chat.completions.create(
            model=self.config['model'],
            messages=messages,
            temperature=1 if self.config['model'] in O_MODELS else 0.4
        )
        return response.choices[0].message.content

    def __max_model_tokens(self):
        base = 4096
        if self.config['model'] in GPT_3_MODELS:
            return base
        if self.config['model'] in GPT_3_16K_MODELS:
            return base * 4
        if self.config['model'] in GPT_4_MODELS:
            return base * 2
        if self.config['model'] in GPT_4_32K_MODELS:
            return base * 8
        if self.config['model'] in GPT_4_VISION_MODELS:
            return base * 31
        if self.config['model'] in GPT_4_128K_MODELS:
            return base * 31
        if self.config['model'] in GPT_4O_MODELS:
            return base * 31
        elif self.config['model'] in O_MODELS:
            # https://platform.openai.com/docs/models#o1
            if self.config['model'] == "o1":
                return 100_000
            elif self.config['model'] == "o1-preview":
                return 32_768
            else:
                return 65_536
        raise NotImplementedError(
            f"Max tokens for model {self.config['model']} is not implemented yet."
        )

    # https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
    def __count_tokens(self, messages) -> int:
        """
        Counts the number of tokens required to send the given messages.
        :param messages: the messages to send
        :return: the number of tokens required
        """
        model = self.config['model']
        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            encoding = tiktoken.get_encoding("o200k_base")

        if model in GPT_ALL_MODELS:
            tokens_per_message = 3
            tokens_per_name = 1
        else:
            raise NotImplementedError(f"""num_tokens_from_messages() is not implemented for model {model}.""")
        num_tokens = 0
        for message in messages:
            num_tokens += tokens_per_message
            for key, value in message.items():
                if key == 'content':
                    if isinstance(value, str):
                        num_tokens += len(encoding.encode(value))
                    else:
                        for message1 in value:
                            if message1['type'] == 'image_url':
                                image = decode_image(message1['image_url']['url'])
                                num_tokens += self.__count_tokens_vision(image)
                            else:
                                num_tokens += len(encoding.encode(message1['text']))
                else:
                    num_tokens += len(encoding.encode(value))
                    if key == "name":
                        num_tokens += tokens_per_name
        num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
        return num_tokens

    # no longer needed

    def __count_tokens_vision(self, image_bytes: bytes) -> int:
        """
        Counts the number of tokens for interpreting an image.
        :param image_bytes: image to interpret
        :return: the number of tokens required
        """
        image_file = io.BytesIO(image_bytes)
        image = Image.open(image_file)
        model = self.config['vision_model']
        if model not in GPT_4_VISION_MODELS:
            raise NotImplementedError(f"""count_tokens_vision() is not implemented for model {model}.""")
        
        w, h = image.size
        if w > h: w, h = h, w
        # this computation follows https://platform.openai.com/docs/guides/vision and https://openai.com/pricing#gpt-4-turbo
        base_tokens = 85
        detail = self.config['vision_detail']
        if detail == 'low':
            return base_tokens
        elif detail == 'high' or detail == 'auto': # assuming worst cost for auto
            f = max(w / 768, h / 2048)
            if f > 1:
                w, h = int(w / f), int(h / f)
            tw, th = (w + 511) // 512, (h + 511) // 512
            tiles = tw * th
            num_tokens = base_tokens + tiles * 170
            return num_tokens
        else:
            raise NotImplementedError(f"""unknown parameter detail={detail} for model {model}.""")

    # No longer works as of July 21st 2023, as OpenAI has removed the billing API
    # def get_billing_current_month(self):
    #     """Gets billed usage for current month from OpenAI API.
    #
    #     :return: dollar amount of usage this month
    #     """
    #     headers = {
    #         "Authorization": f"Bearer {openai.api_key}"
    #     }
    #     # calculate first and last day of current month
    #     today = date.today()
    #     first_day = date(today.year, today.month, 1)
    #     _, last_day_of_month = monthrange(today.year, today.month)
    #     last_day = date(today.year, today.month, last_day_of_month)
    #     params = {
    #         "start_date": first_day,
    #         "end_date": last_day
    #     }
    #     response = requests.get("https://api.openai.com/dashboard/billing/usage", headers=headers, params=params)
    #     billing_data = json.loads(response.text)
    #     usage_month = billing_data["total_usage"] / 100  # convert cent amount to dollars
    #     return usage_month
