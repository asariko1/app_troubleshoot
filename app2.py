#gemi3 version works fine

import streamlit as st
import os
from dotenv import load_dotenv

# --- ADD THIS FOR STREAMLIT DEPLOYMENT ---
# Streamlit uses an old version of SQLite that breaks ChromaDB. 
# This "trick" fixes it.
try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass
# -----------------------------------------

# Your original working imports
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage

# 1. SETUP SECRETS & DATA
load_dotenv()

# Updated to look for Streamlit Secrets OR .env file
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") or st.secrets.get("GOOGLE_API_KEY")
user_phone = os.getenv("USER_PHONE", "Not Provided")
user_email = os.getenv("USER_EMAIL", "Not Provided")
contact_info = f"Phone: {user_phone}\nEmail: {user_email}"

# Ensure API Key is present
if not GOOGLE_API_KEY:
    st.error("Missing GOOGLE_API_KEY! Please add it to Streamlit Secrets.")
    st.stop()

# Load the file
if os.path.exists("helpdesk_guides.txt"):
    with open("helpdesk_guides.txt", "r", encoding="utf-8") as file:
        content = file.read()
else:
    st.error("Missing helpdesk_guides.txt!")
    st.stop()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_text(content)

# 2. SET UP THE ENGINE
@st.cache_resource
def init_rag():
    # Use the 2025 updated embedding model
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004", 
        google_api_key=GOOGLE_API_KEY
    )
    
    # Using your preferred model
    model = ChatGoogleGenerativeAI(
        model="gemini-3-flash-preview", # or "gemini-2.0-flash"
        google_api_key=GOOGLE_API_KEY
    )
    
    # We use a temporary directory for Streamlit Cloud
    persist_dir = "./chroma_db"
    
    vectorstore = Chroma.from_texts(
        texts=chunks, 
        embedding=embeddings, 
        persist_directory=persist_dir
    )
    
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    return model, retriever

model, retriever = init_rag()

# 3. CONTEXT COMBINER
def get_full_context(input_data):
    actual_question = input_data["question"]
    docs = retriever.invoke(actual_question)
    context_text = "\n\n".join(doc.page_content for doc in docs)
    return f"{context_text}\n\nSUPPORT CONTACT:\n{contact_info}"

# 4. UI LAYOUT
st.set_page_config(page_title="Royal App Helpdesk", page_icon="👑")
st.title("👑 Royal App Support Assistant")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for message in st.session_state.chat_history:
    role = "Human" if isinstance(message, HumanMessage) else "AI"
    with st.chat_message(role):
        st.markdown(message.content)

# 5. THE CHAIN
template = """
You are "RoyalBot," the friendly and professional AI Support Agent for Royal App.

# PERSONA:
- You are patient, tech-savvy, and warm. 
- You use emojis (like 👑, 🚀, 🛠️) to keep the tone helpful.
- Your goal is to make the user feel supported, even if their problem is frustrating.

# CONVERSATION GUIDELINES:
1. GREETINGS: If the user says "hi", "hello", or "how are you", respond warmly as RoyalBot. Introduce yourself and ask how you can help with the Royal App today.
2. TROUBLESHOOTING: For ALL technical steps, you MUST ONLY use the 'Context from Guide' provided below. 
3. NO HALLUCINATIONS: If a technical question is NOT covered in the context, say: "I'm sorry, I don't have the specific steps for that in my current manual. Please reach out to our Digital Manager for further help."
4. STRUCTURE: Use bullet points for steps to make them easy to read.

Chat History:
{chat_history}

Context from Guide:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

chain = (
    {
        "context": get_full_context, 
        "question": lambda x: x["question"],
        "chat_history": lambda x: x["chat_history"]
    }
    | prompt | model | StrOutputParser()
)
# 6. EXECUTION
user_query = st.chat_input("How can I help you with the Royal App?")

if user_query:
    with st.chat_message("Human"):
        st.markdown(user_query)
    
    response = chain.invoke({
        "question": user_query, 
        "chat_history": st.session_state.chat_history
    })
    
    with st.chat_message("AI"):
        st.markdown(response)
    
    st.session_state.chat_history.append(HumanMessage(content=user_query))
    st.session_state.chat_history.append(AIMessage(content=response))