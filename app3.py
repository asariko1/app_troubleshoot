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

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage

# --- 2. CONFIG ---
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") or st.secrets.get("GOOGLE_API_KEY")

# --- 3. DATA ENGINE (High Accuracy Settings) ---
@st.cache_resource
def init_rag():
    with open("helpdesk_guides.txt", "r", encoding="utf-8") as file:
        content = file.read()

    # Smaller chunks with more overlap = higher accuracy for specific names like 'GA Tool'
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=150)
    chunks = text_splitter.split_text(content)
    
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=GOOGLE_API_KEY)
    vectorstore = Chroma.from_texts(texts=chunks, embedding=embeddings)
    
    # Switched to Gemini 2.5 Flash for better personality/stability
    model = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", 
        google_api_key=GOOGLE_API_KEY,
        temperature=0.2 # Lower temperature = higher accuracy
    )
    
    # k=7 means it looks at 7 parts of your guide instead of 5
    return model, vectorstore.as_retriever(search_kwargs={"k": 7})

model, retriever = init_rag()

def get_context(input_data):
    docs = retriever.invoke(input_data["question"])
    return "\n\n".join(doc.page_content for doc in docs)

# --- 4. THE ANTI-LOOP TEMPLATE ---
template = """
You are "RoyalBot," the expert Support Agent for Royal App. 👑

# YOUR PERSONA:
- You are not just a bot; you are a tech-savvy concierge. 
- You are warm, encouraging, and professional.
- ALWAYS use emojis (🚀, ✨, 🛠️, 👑) to keep the tone positive and high-end.
- If the user is having a hard time, use empathetic phrases like "I understand how frustrating that can be, let's solve it together!"

# TONE & STYLE:
- Avoid being "robotic" or "flat." 
- Use conversational transitions like "Great question!" or "I'd be happy to help with that."
- Make your bullet points clear but friendly.

# SMART ROUTING:
- Check if the user mentioned their device (iOS/iPhone or Android). 
- If the troubleshooting steps differ and you don't know their device yet, you MUST ask: "I want to make sure I give you the perfect steps! Are you using an iPhone or an Android?"

# NEW: VISUAL AIDS (Rule 6):
- If the 'Context' contains an image link (e.g., ![Icon](url)), you MUST include it in your response to help the user visualize the step. 
- Place the image directly below the text description of that step.



# THE BOLD TRIGGER RULE:
- I have marked tools in *Bold Text*.
- IMPORTANT: Check the 'Chat History' before asking a follow-up.
- DO NOT ask "Would you like me to show you how to update..." IF:
    1. The user just said "Yes", "Sure", or "Please". In this case, just give the instructions!
    2. You already asked this exact question in the very last message.
- ONLY ask the follow-up if it is a brand new topic

# CONVERSATION GUIDELINES:
1. GREETINGS: Respond warmly and introduce yourself as RoyalBot.
2. TROUBLESHOOTING: Use ONLY the 'Context from Guide' below for facts.
3. NO HALLUCINATIONS: Refer to the Digital Manager if the info is missing.


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
st.markdown("I’d be happy to help you navigate the app or solve any technical hiccups you might be experiencing. My goal is to make sure your digital experience is as smooth as a calm sea! 🌊 (H2)")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

  # --- PASTE THE WELCOME LOGIC HERE ---
    welcome_message = (

    "Greetings! 👑 I am RoyalBot, your dedicated tech-savvy concierge for the Royal App! 🚀 I'm here to ensure your digital experience is as smooth as a calm sea and absolutely majestic. ✨ Whether you need help **signing in**, **making reservations**, or getting your **chat to work**, I've got the tools to help! 🛠️"
    "\n\n"
    "To make sure I give you the perfect steps for your journey, are you currently using an **iPhone** or an **Android**? 📱"
)
    
    # Add the welcome message as the first AI entry
    st.session_state.chat_history.append(AIMessage(content=welcome_message))
    # --- END OF WELCOME LOGIC ---



for message in st.session_state.chat_history:
    role = "user" if isinstance(message, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(message.content)

if user_query := st.chat_input("How can I help?"):
    with st.chat_message("user"):
        st.markdown(user_query)
    
    with st.chat_message("assistant"):
        response = chain.invoke({
            "question": user_query,
            "chat_history": st.session_state.chat_history
        })
        st.markdown(response)
    
    st.session_state.chat_history.append(HumanMessage(content=user_query))
    st.session_state.chat_history.append(AIMessage(content=response))