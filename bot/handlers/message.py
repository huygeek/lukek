
from telegram import Update
from telegram.ext import ContextTypes

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Handle incoming messages and respond with a single message
    """
    user_input = update.message.text
    chat_id = update.effective_chat.id

    try:
        # Get complete response from OpenAI without streaming
        response, tokens = await context.bot_data['openai'].get_chat_response(
            chat_id=chat_id,
            query=user_input
        )
        
        # Send single response message
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text=response,
            reply_to_message_id=update.message.message_id
        )
        
    except Exception as e:
        error_text = f"Error: {str(e)}"
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text=error_text,
            reply_to_message_id=update.message.message_id
        )
