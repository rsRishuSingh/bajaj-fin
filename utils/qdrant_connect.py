import os
import uuid
import time
from dotenv import load_dotenv
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer
import warnings

warnings.filterwarnings("ignore", category=FutureWarning, module="torch")

load_dotenv()

embedder = SentenceTransformer(os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"))

qdrant_client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
)

BATCH_SIZE = 256 # Process 128 documents at a time

def create_collection(collection_name):
    """Ensures the Qdrant collection exists with the correct configuration."""
    vector_size = embedder.get_sentence_embedding_dimension()
    try:
        if not qdrant_client.collection_exists(collection_name=collection_name):
            print(f"Creating collection '{collection_name}'...")
            qdrant_client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
            )
            print("Collection created successfully.")
        else:
            print(f"Collection '{collection_name}' already exists.")
        return True
    except Exception as e:
        print(f"Error creating or verifying collection: {e}")
        return False
    
def delete_collection(collection_name):
    """Ensures the Qdrant collection is deleted."""
    vector_size = embedder.get_sentence_embedding_dimension()
    try:
        if qdrant_client.collection_exists(collection_name=collection_name):
            print(f"Deleting collection '{collection_name}'...")
            qdrant_client.delete_collection(collection_name=collection_name)
            print("Collection deleted successfully.")
        else:
            print(f"No collection '{collection_name}' exists.")
        return True
    except Exception as e:
        print(f"Error deleting or verifying collection: {e}")
        return False


def insert_data_in_batches(docs, collection_name)->bool:
    """
    Encodes documents and inserts them into Qdrant in batches to avoid timeouts.
    """
    print(f"Starting data insertion with a batch size of {BATCH_SIZE}...")
    total_docs = len(docs)
    
    # Iterate over the documents in batches
    for i in range(0, total_docs, BATCH_SIZE):
        batch_docs = docs[i : i + BATCH_SIZE]
        start_time = time.time()
        
        print(f"Processing batch {i//BATCH_SIZE + 1}/{(total_docs + BATCH_SIZE - 1)//BATCH_SIZE}...")

        embeddings = embedder.encode(
            [doc.page_content for doc in batch_docs],
            show_progress_bar=False  # Disable nested progress bar
        )


        points = [
            models.PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding.tolist(),
                payload={**doc.metadata, "text": doc.page_content}
            )
            for doc, embedding in zip(batch_docs, embeddings)
        ]


        try:
            qdrant_client.upsert(
                collection_name=collection_name,
                points=points,
                wait=True  # Wait for the operation to complete for this batch
            )
            end_time = time.time()
            print(f"  - Batch uploaded successfully in {end_time - start_time:.2f} seconds.")

        except Exception as e:
            print(f"  - Error uploading batch {i//BATCH_SIZE + 1}: {e}")
            print("  - Skipping this batch and continuing...")
            return False
            
        #small delay to avoid overwhelming the server
        time.sleep(0.05)

    print("All data has been processed.")
    return True


    