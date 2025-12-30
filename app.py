import streamlit as st
import os
from dotenv import load_dotenv

# 1. IMPORTS (The "Guest List")
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage

# Define our preferred working model here
WORKING_MODEL_NAME = "gemini-3-flash-preview" # Or whatever name you confirmed works!

# 2. SETUP SECRETS & DATA
load_dotenv()
user_phone = os.getenv("USER_PHONE")
user_email = os.getenv("USER_EMAIL")
contact_info = f"Phone: {user_phone}\nEmail: {user_email}"

# Load the CV text file immediately so the whole script can see it
if os.path.exists("helpdesk_guides.txt"):
    # Add 'encoding="utf-8"' to tell Python exactly how to read the special characters
    with open("helpdesk_guides.txt", "r", encoding="utf-8") as file:
        content = file.read()
else:
    st.error("Missing helpdesk_guides.txt! Please create it in your project folder.")
    st.stop()

# Prepare the chunks (the "pages" for our librarian)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_text(content)

# 3. SET UP THE ENGINE (Cached for performance)
@st.cache_resource
def init_rag():
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    # We use the stable 2.5-flash to avoid the "Limit 0" errors
    model = ChatGoogleGenerativeAI(model=WORKING_MODEL_NAME)
    
    persist_dir = "helpdek_database"
    
    # Check if we need to build a new database or load an old one
    if os.path.exists(persist_dir):
        vectorstore = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
    else:
        vectorstore = Chroma.from_texts(texts=chunks, embedding=embeddings, persist_directory=persist_dir)
    
    retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
    return model, retriever

model, retriever = init_rag()

# 4. CONTEXT COMBINER
def get_full_context(input_data):
    actual_question = input_data["question"]
    docs = retriever.invoke(actual_question)
    cv_text = "\n\n".join(doc.page_content for doc in docs)
    return f"{cv_text}\n\nPRIVATE CONTACT INFO:\n{contact_info}"

# 5. UI LAYOUT
st.set_page_config(page_title="App Helpdesk", page_icon="🚀")
st.title("🤖 Chat with Royal App's AI Bot")

# Initialize Chat History in Session State
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display previous messages
for message in st.session_state.chat_history:
    if isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.markdown(message.content)
    else:
        with st.chat_message("AI"):
            st.markdown(message.content)

# 6. THE CONVEYOR BELT (Chain)
template = """
You are the personal AI Helpdesk Agent for Royal Caribbean. 

You are an expert, patient, and professional IT Helpdesk Agent for this company.
Your goal is to solve the user's problem by using ONLY the provided context.

STRICT RULES:
1.  *FACTS ONLY:* For ALL troubleshooting steps, you MUST ONLY use the 'Context' provided below.
2.  *NO GUESSING:* If the 'Context' does not have the answer, you MUST reply: 'I am sorry, that specific guide has not been loaded yet. Please call the Digital Manager.'
3.  *ALWAYS BE HELPFUL:* Use a polite and professional tone.

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

# 7. CHAT INPUT & EXECUTION
user_query = st.chat_input("Ask me about Asar's experience...")

if user_query:
    # Show user message immediately
    with st.chat_message("Human"):
        st.markdown(user_query)
    
    # Run the Chain
    response = chain.invoke({
        "question": user_query, 
        "chat_history": st.session_state.chat_history
    })
    
    # Show AI response
    with st.chat_message("AI"):
        st.markdown(response)
    
    # Add to history
    st.session_state.chat_history.append(HumanMessage(content=user_query))
    st.session_state.chat_history.append(AIMessage(content=response))

# 8. THE SPY HOLE (Debug Mode)
with st.sidebar:
    st.divider()
    with st.expander("🕵️ Debug: Librarian's Search Results"):
        if user_query:
            debug_info = get_full_context({"question": user_query})
            st.write(debug_info)
        else:
            st.write("Waiting for a question...")
            