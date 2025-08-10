import os
import sys
import warnings
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings

# CONFIGURATION 
load_dotenv()
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")
sys.setrecursionlimit(10**5)

# AZURE & QDRANT CREDENTIALS
AZURE_OPENAI_CHAT_ENDPOINT = os.getenv("AZURE_OPENAI_CHAT_ENDPOINT")
AZURE_OPENAI_CHAT_API_KEY = os.getenv("AZURE_OPENAI_CHAT_API_KEY")
AZURE_OPENAI_EMB_ENDPOINT = os.getenv("AZURE_OPENAI_EMB_ENDPOINT")
AZURE_OPENAI_EMB_API_KEY = os.getenv("AZURE_OPENAI_EMB_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

# SHARED LLM & EMBEDDER INSTANCES
LLM = AzureChatOpenAI(
    deployment_name="gpt-4.1-mini",
    azure_endpoint=AZURE_OPENAI_CHAT_ENDPOINT,
    api_version="2024-05-01-preview",
    api_key=AZURE_OPENAI_CHAT_API_KEY,
    temperature=0,
    request_timeout=60
)

EMBEDDER = AzureOpenAIEmbeddings(
    azure_deployment="text-embedding-ada-002",
    openai_api_version="2023-05-15",
    azure_endpoint=AZURE_OPENAI_EMB_ENDPOINT,
    api_key=AZURE_OPENAI_EMB_API_KEY,
)