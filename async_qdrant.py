import os
import uuid
import time
import asyncio
from dotenv import load_dotenv
from qdrant_client import AsyncQdrantClient, models
from sentence_transformers import SentenceTransformer
import warnings

warnings.filterwarnings("ignore", category=FutureWarning, module="torch")

load_dotenv()

# Initialize embedder and async Qdrant client
embedder = SentenceTransformer(os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"))
async_client = AsyncQdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
)

BATCH_SIZE = 36     # Number of docs per batch
MAX_CONCURRENT = 6    # Number of parallel upsert tasks

async def create_collection(collection_name: str) -> bool:
    """Ensure the Qdrant collection exists with the correct configuration."""
    vector_size = embedder.get_sentence_embedding_dimension()
    try:
        # Try retrieving collection info; exception if not exists
        await async_client.get_collection(collection_name=collection_name)
        print(f"Collection '{collection_name}' already exists.")
    except Exception as e:
        print(f"Creating collection '{collection_name}'...")
        await async_client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=vector_size,
                distance=models.Distance.COSINE
            ),
        )
        print("Collection created successfully.")
    return True

async def delete_collection(collection_name: str) -> bool:
    """Ensure the Qdrant collection is deleted."""
    try:
        print(f"Deleting collection '{collection_name}' if it exists...")
        await async_client.delete_collection(collection_name=collection_name)
        print("Collection deleted successfully.")
        return True
    except Exception as e:
        print(f"Error deleting collection: {e}")
        return False

async def _upsert_batch(collection_name: str, batch_docs) -> None:
    """Encode and upsert a single batch of documents asynchronously."""
    embeddings = embedder.encode(
        [doc.page_content for doc in batch_docs],
        show_progress_bar=False
    )
    points = [
        models.PointStruct(
            id=str(uuid.uuid4()),
            vector=emb.tolist(),
            payload={**doc.metadata, "text": doc.page_content}
        )
        for doc, emb in zip(batch_docs, embeddings)
    ]
    await async_client.upsert(
        collection_name=collection_name,
        points=points,
        wait=True
    )

async def insert_data_parallel(docs, collection_name: str) -> bool:
    """
    Inserts documents into Qdrant in parallel batches using asyncio to avoid write timeouts.
    """
    total = len(docs)
    total_batches = (total + BATCH_SIZE - 1) // BATCH_SIZE
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    tasks = []

    print(f"Starting parallel insertion: {total_batches} batches of up to {BATCH_SIZE} docs (max {MAX_CONCURRENT} concurrent)")

    for idx in range(total_batches):
        start = idx * BATCH_SIZE
        batch = docs[start : start + BATCH_SIZE]
        batch_idx = idx + 1

        async def worker(batch_docs, batch_idx=batch_idx):
            async with semaphore:
                t0 = time.perf_counter()
                print(f"Processing batch {batch_idx}/{total_batches}...")
                try:
                    await _upsert_batch(collection_name, batch_docs)
                    elapsed = time.perf_counter() - t0
                    print(f"  - Batch {batch_idx} uploaded in {elapsed:.2f}s")
                except Exception as e:
                    print(f"  - Error in batch {batch_idx}: {e}")
                    raise

        tasks.append(worker(batch))

    try:
        await asyncio.gather(*tasks)
        print("All data has been processed.")
        return True
    except Exception as e:
        print("Insertion terminated due to errors.",e)
        return False

# Run example:
# import asyncio
# docs = [...]  # list of your Document-like objects
# asyncio.run(create_collection("my_collection"))
# success = asyncio.run(insert_data_parallel(docs, "my_collection"))
# print("Insertion successful?", success)
# asyncio.run(delete_collection("my_collection"))
