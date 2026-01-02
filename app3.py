import os
import streamlit as st
from dotenv import load_dotenv

# --- 1. DEPLOYMENT FIX ---
try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

import csv
import pandas as pd
from datetime import datetime
from tenacity import retry_if_exception_type
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage

# --- LOGGING & KEYWORDS ---
# Unified keyword list and function to log gaps
FAIL_KEYWORDS = ["Digital Manager", "I don't know", "not in the guide", "couldn't find", "contact support"]

# Words you want to always appear in **Bold**
BOLD_KEYWORDS = ["Sign In", "Reservation", "Digital Manager", "Settings", "iPhone", "Android", "VPN", "MAC", "Limit IP Adress", "Private WIfi", "GA Tool", "Private Relay"]

def bold_important_words(text):
    for word in BOLD_KEYWORDS:
        # This replaces the word with a bold version (e.g., "Sign In" -> "*Sign In*")
        # .replace is case-sensitive, so ensure your list matches or use regex for advanced matching
        text = text.replace(word, f"**{word}**")
    return text


def log_unanswered_question(question, response):
    # This creates (or appends to) a file called 'bot_gaps.csv'
    file_path = 'bot_gaps.csv'
    file_exists = os.path.isfile(file_path)
    with open(file_path, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(['Timestamp', 'User Question', 'Bot Response'])
        writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), question, response])

# --- 2. CONFIG ---
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") or st.secrets.get("GOOGLE_API_KEY")

# --- 3. DATA ENGINE (High Accuracy Settings) ---
@st.cache_resource
def init_rag():
    with open("helpdesk_guides.txt", "r", encoding="utf-8") as file:
        content = file.read()

    # Smaller chunks with more overlap = higher accuracy
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=150)
    chunks = text_splitter.split_text(content)
    
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=GOOGLE_API_KEY)
    vectorstore = Chroma.from_texts(texts=chunks, embedding=embeddings)
    
    # Updated to latest stable Gemini 3 Flash (Dec 2025)
    # Temperature 0.0 makes him straight to the point
    base_model = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", 
        google_api_key=GOOGLE_API_KEY,
        temperature=0.0 
    )
    
    model = base_model.with_retry(
        stop_after_attempt=3,
        wait_exponential_jitter=True 
    )
    
    return model, vectorstore.as_retriever(search_kwargs={"k": 7})

model, retriever = init_rag()

# --- SIDEBAR LOG VIEWER ---
with st.sidebar:
    st.title("🛠️ RoyalBot Dev Tools")
    if st.checkbox("Show Knowledge Gaps"):
        st.write("Questions missing from your guide:")
        if os.path.exists('bot_gaps.csv'):
            df_logs = pd.read_csv('bot_gaps.csv')
            st.dataframe(df_logs.iloc[::-1], use_container_width=True) # Newest first
            
            with open('bot_gaps.csv', 'rb') as f:
                st.download_button(
                    label="📥 Download CSV Logs",
                    data=f,
                    file_name=f"royal_gaps_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime='text/csv'
                )
        else:
            st.info("No gaps recorded yet. 👑")

def get_context(input_data):
    docs = retriever.invoke(input_data["question"])
    return "\n\n".join(doc.page_content for doc in docs)

# --- 4. THE CONCISE TECHNICAL TEMPLATE ---
template = """
You are "RoyalBot," the expert Technical Support Specialist for Royal App. 👑

# OBJECTIVE:
- Provide direct, technical, and extremely concise solutions.
- Only ask about the user's device (iPhone/Android) IF the troubleshooting steps in the 'Context' are different for each platform.
- If the steps are the same for both, just provide the answer immediately.

# RULES:
1. BREVITY: Keep responses under 4 sentences unless listing complex steps.
2. SOURCE ONLY: Use ONLY the 'Context' below. If info is missing, say: "Information not found. Please contact the Digital Manager."
3. DEVICE CHECK: If steps differ for iOS/Android and you don't know their device, ask: "Are you on iPhone or Android?"
4. NO FOLLOW-UPS: Never ask "Is there anything else?" or "Would you like me to show you...". Just answer and stop.
5. VISUALS: If 'Context' has an image link (e.g. ![Icon](url)), include it directly below the text for that step.





Chat History:
{chat_history}

Context:
{context}

Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)

chain = (
    {
        "context": get_context, 
        "question": lambda x: x["question"],
        "chat_history": lambda x: x["chat_history"]
    }
    | prompt | model | StrOutputParser()
)

# --- 5. EXECUTION ---
st.title("Royal App AI Support")
st.markdown("### Technical support at your fingertips. 🌊")

# --- WELCOME LOGIC (FIXED: Only appends once) ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
    welcome_message = (
        "Greetings! 👑 I am RoyalBot, your technical concierge. 🚀 I can help with **signing in**, **reservations**, or **chat issues**. 🛠️"
        "\n\n"
        
    )
    st.session_state.chat_history.append(AIMessage(content=welcome_message))

# Display history
for message in st.session_state.chat_history:
    role = "user" if isinstance(message, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(message.content)

# Chat Input & Logic
if user_query := st.chat_input("How can I help?"):
    with st.chat_message("user"):
        st.markdown(user_query)
    
    with st.chat_message("assistant"):
        with st.spinner("RoyalBot is thinking... 👑"):
            try:
                response = chain.invoke({
                    "question": user_query,
                    "chat_history": st.session_state.chat_history
                })

# --- APPLY BOLDING HERE ---
                response = bold_important_words(response)

                st.markdown(response)

                # NEW LOGGING LOGIC
                if any(k.lower() in response.lower() for k in FAIL_KEYWORDS):
                    
                    # Save it to history FIRST so the user sees it after the rerun
                    st.session_state.chat_history.append(HumanMessage(content=user_query))
                    st.session_state.chat_history.append(AIMessage(content=response))
                    # than log to cvs file

                    log_unanswered_question(user_query, response)
                    st.rerun()

            except Exception as e:
                if "429" in str(e):
                    response = "System busy. 👑 Please wait a moment."
                else:
                    response = "I encountered a little hiccup. 🛠️"
                st.markdown(response)

    # Update History
    st.session_state.chat_history.append(HumanMessage(content=user_query))
    st.session_state.chat_history.append(AIMessage(content=response))