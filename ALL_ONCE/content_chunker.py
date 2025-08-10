from typing import List, Dict
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi

from text_extractor import extract_text
from file_downloader import run_download_and_identify_file

def chunk_text_with_metadata(full_text: str, pages_text: List[Dict]) -> List[Dict]:
    """Performs text splitting and attaches page number metadata."""
    print("🔪 Performing semantic chunking...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=768, chunk_overlap=128)
    documents = splitter.create_documents([full_text])
    
    chunk_contents = [doc.page_content for doc in documents]
    chunks_with_metadata = []
    
    char_offset = 0
    page_map = [{'offset': 0, 'page': 1}]
    for page in pages_text:
        char_offset += len(page['text']) + 2
        page_map.append({"offset": char_offset, "page": page['page']})
    
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

if __name__ == "__main___":
    pdf_url = "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D"
    file_content, file_type = run_download_and_identify_file(pdf_url)

    content = extract_text(file_content, file_type)
    chunks = chunk_text_with_metadata(content)
    print("chunks")
    print(chunks[:5])