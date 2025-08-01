import os
from dotenv import load_dotenv
from openai import AzureOpenAI  # From OpenAI's SDK

load_dotenv()

endpoint = "https://rishu.openai.azure.com/"
api_version = "2024-05-01-preview"
deployment = "text-embedding-3-small"

api_key = os.getenv("AZURE_OPENAI_EMBEDDING_API")


# ✅ Use api_key directly 
client = AzureOpenAI(
    api_key=api_key,
    api_version=api_version,
    azure_endpoint=endpoint
)

response = client.embeddings.create(
    input=["first phrase", "second phrase", "third phrase"],
    model=deployment
)

for item in response.data:
    length = len(item.embedding)
    print(
        f"data[{item.index}]: length={length}, "
        f"[{item.embedding[0]}, {item.embedding[1]}, "
        f"..., {item.embedding[-2]}, {item.embedding[-1]}]"
    )

print(response.usage)
