from typing import List, Dict
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi


from text_extractor import extract_text
from file_downloader import run_download_and_identify_file
from qdrant_ops import get_qdrant_collection, upsert_chunks_qdrant
from config import get_embedding_client, get_qdrant_clients

def chunk_text_with_metadata(pages_text: List[Dict]) -> List[Dict]:
    """
    Performs text splitting and attaches page number metadata.
    It now only requires the page-by-page text.
    """
    print("🔪 Performing semantic chunking...")
    
    # 1. full_text is now created inside the function
    full_text = "\n\n".join(p['text'] for p in pages_text)
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=768, chunk_overlap=128)
    documents = splitter.create_documents([full_text])
    
    chunk_contents = [doc.page_content for doc in documents]
    chunks_with_metadata = []
    
    # Create a map of character offsets to page numbers
    char_offset = 0
    page_map = [{'offset': 0, 'page': 1}]
    # The `+ 2` here corresponds to the length of our separator: `\n\n`
    for page in pages_text:
        char_offset += len(page['text']) + 2 
        page_map.append({"offset": char_offset, "page": page['page']})
    
    # Find the page number for each chunk
    for content in chunk_contents:
        if len(content.strip()) < 50: continue
        try:
            start_index = full_text.find(content)
            if start_index == -1: continue

            page_num = next(p['page'] for p in reversed(page_map) if start_index >= p['offset'])
            chunks_with_metadata.append({"text": content, "page": page_num})
        except StopIteration:
            chunks_with_metadata.append({"text": content, "page": "N/A"})
            
    return chunks_with_metadata

def initialize_bm25(chunk_texts: List[str]) -> BM25Okapi:
    """Initializes the BM25 retriever."""
    print("🔍 Initializing BM25 retriever...")
    tokenized_corpus = [doc.lower().split() for doc in chunk_texts]
    return BM25Okapi(tokenized_corpus)


# ==================================
# MAIN EXECUTION BLOCK
# ==================================
# All imports should be at the top of the file
import asyncio
# ... (other imports like get_qdrant_clients, get_embedding_client, etc.)

async def main():
    """
    Main asynchronous function to run the entire indexing pipeline.
    """
    pdf_url = "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D"
    
    # 1. Ingestion and Processing
    file_content, file_type = await run_download_and_identify_file(pdf_url)
    pages = extract_text(file_content, file_type)
    chunks = chunk_text_with_metadata(pages)
    
    print("\n✅ Chunking complete. Here are the first 2 chunks with metadata:")
    for i, chunk in enumerate(chunks[:2]):
        print(f"\n--- Chunk {i+1} | Page: {chunk['page']} ---")
        print(chunk['text'])
        
    # 2. Database and AI Model Setup
    sync_client, async_client = get_qdrant_clients()
    embedder = get_embedding_client()
    get_qdrant_collection(sync_client, "rishu_demo_1", embedder)

    # 3. Indexing (Simplified and corrected)
    print("\n🚀 Upserting chunks to Qdrant...")
    # Directly await the function. It's cleaner and achieves the same result.
    await upsert_chunks_qdrant(
        async_client, "rishu_demo_1", chunks, embedder, pdf_url
    )
    print("✅ Indexing complete.")

if __name__ == "__main__":
    # This is the standard way to run a top-level async function.
    asyncio.run(main())