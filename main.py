import os
import asyncio
import random
import logging
from dotenv import load_dotenv
from telegram import Update, ReactionTypeEmoji
from telegram.ext import ApplicationBuilder, ContextTypes, MessageHandler, filters, AIORateLimiter

# LangChain & Gemini Imports
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories.upstash_redis import UpstashRedisChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# 1. Setup
load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 2. Load Umida's Brain (Gemini version)
# Ensure this model matches the one you used in Colab to build the brain
# In section 2 of your main.py on GitHub
embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
vectorstore = Chroma(persist_directory="./umida_brain_db", embedding_function=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# Use Gemini 1.5 Flash for the fastest response times
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)

# 3. Chain Logic
system_prompt = (
    "Sizning ismingiz Umida. Siz professional psixologsiz. "
    "Foydalanuvchiga faqat O'ZBEK tilida, hamdardlik bilan javob bering.\n\n"
    "Context:\n{context}"
)

contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", "Suhbat tarixini hisobga olib, savolni mustaqil holga keltiring."),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

# 4. Redis Memory Manager
def get_session_history(session_id: str):
    return UpstashRedisChatMessageHistory(
        url=os.environ["UPSTASH_REDIS_REST_URL"],
        token=os.environ["UPSTASH_REDIS_REST_TOKEN"],
        session_id=session_id,
        ttl=2592000 # Memory lasts 30 days
    )

umida_bot = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)

# 5. Telegram Message Handler
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.effective_message or not update.effective_message.text:
        return

    chat_id = str(update.effective_chat.id)
    user_text = update.effective_message.text

    try:
        # 🎭 Reaction
        await update.effective_message.set_reaction(reaction=[ReactionTypeEmoji("🫂")])
        
        # 🧠 Query
        response = await asyncio.to_thread(
            umida_bot.invoke,
            {"input": user_text},
            config={"configurable": {"session_id": chat_id}}
        )
        
        await update.effective_message.reply_text(response['answer'].strip())
        
    except Exception as e:
        logging.error(f"Error in chat {chat_id}: {e}")
        await update.effective_message.reply_text("Kechirasiz, menda juda zarur ish chiqib qoldi. Iltimos, keyinroq gaplashaylik.")

# 6. Entry Point
def main():
    application = ApplicationBuilder().token(os.environ["TELEGRAM_TOKEN"]).rate_limiter(AIORateLimiter()).build()
    application.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_message))
    
    logging.info("🚀 Umida (Gemini Engine) is LIVE!")
    application.run_polling(drop_pending_updates=True)

if __name__ == "__main__":
    main()
