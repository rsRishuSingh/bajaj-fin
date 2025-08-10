# qdrant_ops.py
# Description: Functions for interacting with the Qdrant vector database.

from typing import List, Dict
from qdrant_client import QdrantClient, AsyncQdrantClient, models
from langchain_openai import AzureOpenAIEmbeddings

def get_qdrant_collection(sync_client: QdrantClient, collection_name: str, embedder: AzureOpenAIEmbeddings):
    """Checks if a collection exists in Qdrant and creates it if not."""
    try:
        all_collections = sync_client.get_collections().collections
        collection_names = [collection.name for collection in all_collections]
        if collection_name not in collection_names:
            print(f"   - Collection '{collection_name}' not found. Creating it now...")
            embedding_size = len(embedder.embed_query("test"))
            sync_client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(size=embedding_size, distance=models.Distance.COSINE),
            )
            print("   - ✅ Collection created successfully.")
        else:
            print(f"   - ✅ Collection '{collection_name}' already exists.")
    except Exception as e:
        print(f"   - 🚨 CRITICAL ERROR during Qdrant collection check/creation: {e}")
        raise

async def upsert_chunks_qdrant(
    async_client: AsyncQdrantClient,
    collection_name: str,
    chunks: List[Dict],
    embedder: AzureOpenAIEmbeddings,
    doc_url: str
):
    """Embeds and uploads document chunks to a Qdrant collection."""
    print("   - 🧠 Embedding and uploading chunks to Qdrant...")
    batch_size = 64
    for i in range(0, len(chunks), batch_size):
        batch_chunks = chunks[i:i + batch_size]
        batch_contents = [c['text'] for c in batch_chunks]
        batch_embeddings = await embedder.aembed_documents(batch_contents)
        
        await async_client.upsert(
            collection_name=collection_name,
            points=models.Batch(
                ids=list(range(i, i + len(batch_chunks))),
                vectors=batch_embeddings,
                payloads=[
                    {"text": c['text'], "page": c.get('page', 'N/A'), "source_url": doc_url}
                    for c in batch_chunks
                ]
            ),
            wait=True
        )