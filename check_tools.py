try:
    import langchain
    import langchain_google_genai
    import dotenv
    import chromadb
    print("✅ All systems go! Your AI tools are installed correctly.")
except ImportError as e:
    print(f"❌ Something is missing: {e}")