import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from qdrant_client import QdrantClient, AsyncQdrantClient

load_dotenv()

# Azure OpenAI API Configurations
AZURE_OPENAI_CHAT_ENDPOINT = os.getenv("AZURE_OPENAI_CHAT_ENDPOINT")
AZURE_OPENAI_CHAT_API = os.getenv("AZURE_OPENAI_CHAT_API_KEY")
AZURE_OPENAI_EMB_ENDPOINT = os.getenv("AZURE_OPENAI_EMB_ENDPOINT")
AZURE_OPENAI_EMB_API = os.getenv("AZURE_OPENAI_EMB_API_KEY")

# Qdrant Configurations 
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

# Document URL Configurations 
NEWTON_BOOK_URL = os.getenv("NEWTON_BOOK")

# --- Data Directory ---
DATA_DIR = "rag_data_library"
os.makedirs(DATA_DIR, exist_ok=True)


# --- Client Initializations ---

def get_llm_client():
    """Initializes and returns the Azure Chat LLM client."""
    return AzureChatOpenAI(
        deployment_name="gpt-4.1-mini",
        azure_endpoint=AZURE_OPENAI_CHAT_ENDPOINT,
        api_version="2024-12-01-preview",
        api_key=AZURE_OPENAI_CHAT_API,
        temperature=0,
        request_timeout=60
    )

def get_embedding_client():
    """Initializes and returns the Azure Embeddings client."""
    return AzureOpenAIEmbeddings(
        azure_deployment="text-embedding-ada-002",
        openai_api_version="2023-05-15",
        azure_endpoint=AZURE_OPENAI_EMB_ENDPOINT,
        api_key=AZURE_OPENAI_EMB_API,
    )

def get_qdrant_clients():
    """Initializes and returns both synchronous and asynchronous Qdrant clients."""
    print("☁️ Connecting to Qdrant Cloud...")
    sync_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=60)
    async_client = AsyncQdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=60)
    return sync_client, async_client