import os
from dotenv import load_dotenv
from openai import AzureOpenAI

load_dotenv()

endpoint = "https://rishu-mdodjz43-eastus2.cognitiveservices.azure.com/"
model_name = "gpt-4o"
deployment = "gpt-4o"


api_version = "2024-05-01-preview"
AZURE_OPENAI_GPT_API = os.getenv("AZURE_OPENAI_GPT_API")

client = AzureOpenAI(
    api_version=api_version,
    azure_endpoint=endpoint,
    api_key=AZURE_OPENAI_GPT_API,
)


def generate_answer(query: str, context: str):
    """
    Generates an answer using the LLM based on the query and retrieved context.
    """
    print("🤖 Generating answer with LLM...")
    
    # The system prompt instructs the LLM how to behave
    system_prompt = (
        "You are a helpful RAG assistant. Your task is to answer the user's question "
        "based *only* on the provided context. If the context does not contain the "
        "answer, you must state that you cannot answer based on the provided information."
        "answer to the point while minimize response time and token usage"
    )

    # The user message contains the original query and the context
    user_message_content = f"Context:\n---\n{context}\n---\nQuestion: {query}"

    response = client.chat.completions.create(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message_content},
        ],
        temperature=0.7,
        max_tokens=150,
        top_p=1.0,
        model=deployment
    )
    
    print(response.choices[0].message.content)

