import os
from dotenv import load_dotenv

# 1) Use the community package for AzureChatOpenAI
from langchain_community.chat_models import AzureChatOpenAI

# 2) Import LLMChain and PromptTemplate from their dedicated modules
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate

load_dotenv()  # loads AZURE_OPENAI_API_KEY from .env

# Azure OpenAI configuration
AZURE_ENDPOINT        = "https://rishu-mdodjz43-eastus2.cognitiveservices.azure.com/"
AZURE_DEPLOYMENT_NAME = "gpt-4o"
AZURE_API_VERSION     = "2024-05-01-preview"
AZURE_API_KEY         = os.getenv("AZURE_OPENAI_GPT_API")

# Initialize the AzureChatOpenAI client without deprecated params
llm = AzureChatOpenAI(
    deployment_name=AZURE_DEPLOYMENT_NAME,
    azure_endpoint=AZURE_ENDPOINT,
    api_version=AZURE_API_VERSION,
    api_key=AZURE_API_KEY,
    temperature=0.7,
    max_tokens=150,
)

# Define your prompt template
_template = """
You are a helpful RAG assistant. Your task is to answer the user's question
*only* using the provided context. If the context doesn't contain the answer,
state that you cannot answer based on the provided information.

Context:
---
{context}
---

Question: {query}

Answer:
"""
prompt = PromptTemplate.from_template(_template)

# Wire up the chain
chain = LLMChain(llm=llm, prompt=prompt)

def generate_answer(query: str, context: str) -> str:
    """
    Generates an answer using the LLM based on the query and retrieved context.
    """
    print("🤖 Generating answer with LangChain…")
    # use invoke() instead of deprecated run()
    return chain.invoke({"query": query, "context": context})

if __name__ == "__main__":
    sample_context = "The capital of France is Paris."
    answer = generate_answer("What is the capital of France?", sample_context)
    print("Model answered:", answer)
