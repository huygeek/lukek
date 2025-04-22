import openai

class OpenAIWrapper:
    def __init__(self, config):
        self.api_key = config['openai_api_key']
        self.model = config.get('openai_model', 'gpt-3.5-turbo')

    async def get_chat_response(self, chat_id, query):
        try:
            openai.api_key = self.api_key

            response = await openai.ChatCompletion.acreate(
                model=self.model,
                messages=[{"role": "user", "content": query}],
                temperature=0.7,
                max_tokens=1024,
                stream=False  # Đảm bảo không sử dụng streaming
            )

            content = response['choices'][0]['message']['content']
            total_tokens = response['usage']['total_tokens']
            return content.strip(), total_tokens

        except Exception as e:
            return f"Lỗi khi gọi OpenAI: {str(e)}", 0
