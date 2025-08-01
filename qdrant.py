import os
import uuid
import time
from dotenv import load_dotenv
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer
from chunking import load_docs, save_docs, extract_chunks_from_pdf
from answer import generate_answer

load_dotenv()
MODEL_NAME = "all-MiniLM-L6-v2"
COLLECTION_NAME = "my_document_store_15"

model = SentenceTransformer(MODEL_NAME)


qdrant_client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API"),
)

BATCH_SIZE = 128 # Process 128 documents at a time

def create_collection(COLLECTION_NAME):
    """Ensures the Qdrant collection exists with the correct configuration."""
    vector_size = model.get_sentence_embedding_dimension()
    try:
        if not qdrant_client.collection_exists(collection_name=COLLECTION_NAME):
            print(f"Creating collection '{COLLECTION_NAME}'...")
            qdrant_client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
            )
            print("Collection created successfully.")
        else:
            print(f"Collection '{COLLECTION_NAME}' already exists.")
        return True
    except Exception as e:
        print(f"Error creating or verifying collection: {e}")
        return False

def insert_data_in_batches(docs, COLLECTION_NAME):
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

        embeddings = model.encode(
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
                collection_name=COLLECTION_NAME,
                points=points,
                wait=True  # Wait for the operation to complete for this batch
            )
            end_time = time.time()
            print(f"  - Batch uploaded successfully in {end_time - start_time:.2f} seconds.")

        except Exception as e:
            print(f"  - Error uploading batch {i//BATCH_SIZE + 1}: {e}")
            print("  - Skipping this batch and continuing...")
            continue # Or use 'break' to stop the entire process
            
        #small delay to avoid overwhelming the server
        time.sleep(0.25)

    print("All data has been processed.")

def search_similar_chunks(query: str, COLLECTION_NAME:str, top_k: int = 5):
    """Searches for chunks similar to the query using the modern API."""
    print(f"\n🔍 Searching for: '{query}'")
    
    # 1. Convert the query text to a vector
    query_vector = model.encode(query).tolist()
    
    # 2. Use the 'query_points' method with the 'query' parameter
    search_result = qdrant_client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=top_k
    )
    
    contexts = [point.payload['text'] for point in search_result.points]
        
    if not contexts:
            return "No relevant context was found."

    # Join the individual context strings with a separator
    return "\n---\n".join(contexts)

if __name__ == "__main__":

    print("Started")

    begin = time.time()
    start = time.time()
    docs = extract_chunks_from_pdf("Policy.pdf")
    print("Time Taken in Chunking: ", time.time()-start)

    start = time.time()
    create_collection(COLLECTION_NAME)
    print("Time Taken in Creating database: ",time.time()-start)

    start = time.time()
    insert_data_in_batches(docs, COLLECTION_NAME)
    print("Time Taken in inserting chunks: ",time.time()-start)
    
    
    list_of_questions = [
            "which documents are required to apply for a claim?",
            "How many types of vaccination are available for children of age group between one to twelve years?",
            "What is the name and address of company providing insurance ?",
            "What is the grace period for premium payment under the National Parivar Mediclaim Plus Policy?",
            "What is the waiting period for pre-existing diseases (PED) to be covered?",
            "Does this policy cover maternity expenses, and what are the conditions?",
            "What is the waiting period for cataract surgery?",
            "Are the medical expenses for an organ donor covered under this policy?",
            "What is the No Claim Discount (NCD) offered in this policy?",
            "Is there a benefit for preventive health check-ups?",
            "How does the policy define a 'Hospital'?",
            "What is the extent of coverage for AYUSH treatments?",
            "Are there any sub-limits on room rent and ICU charges for Plan A?"
    ]

    start = time.time()
    for query in list_of_questions:
        context = search_similar_chunks(query, COLLECTION_NAME)
        generate_answer(query, context)
    
    print("Time Taken in answering all the query: ",time.time()-start)
    print("total time: ",time.time()-begin)

    