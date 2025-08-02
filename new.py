import os
import json
import uuid
import requests
from typing import List, Dict, Any, TypedDict

# --- Library Imports ---
from dotenv import load_dotenv
import pandas as pd
import pdfplumber
from langchain.docstore.document import Document
from langchain_openai import AzureOpenAIEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
from langchain.text_splitter import RecursiveCharacterTextSplitter
from chunking import save_docs
import time
# --- Global Configuration ---
load_dotenv()
COLLECTION_NAME = "my_final_policy_documents"
AZURE_ADA_002_VECTOR_SIZE = 1536
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
EMBEDDINGS_MODEL_NAME = "text-embedding-ada-002"
AZURE_EMBEDDING_ENDPOINT = os.getenv('AZURE_OPENAI_EMBEDDING_URL')
AZURE_EMBEDDING_API = os.getenv('AZURE_OPENAI_EMBEDDING_API')

# ==============================================================================
# PART 1: DOCUMENT PROCESSOR
# ==============================================================================

def download_pdf(url: str, save_path: str = "temp.pdf") -> bool:
    """Downloads a PDF from a URL and saves it locally."""
    try:
        print(f"⬇️  Downloading PDF from {url}...")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192): f.write(chunk)
        print(f"✅ PDF saved successfully to '{save_path}'")
        return True
    except requests.exceptions.RequestException as e:
        print(f"❌ Failed to download PDF: {e}")
        return False


def process_pdf(pdf_path: str = "temp.pdf") -> List[Dict[str, Any]]:
    """Extracts text and tables from all pages of a PDF using pdfplumber."""
    if not os.path.exists(pdf_path):
        print(f"❌ File not found at '{pdf_path}'")
        return []
    print("-" * 50)
    
    print(f"Processing '{pdf_path}' with pdfplumber...")
    all_content = []
    
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            text = page.extract_text()
            
            if text and text.strip():
                all_content.append({"page_number": page_num, "type": "text", "content": text})
            
            tables = page.extract_tables()
            
            for table_num, table in enumerate(tables, start=1):
               
                if not table: continue
                df = pd.DataFrame(table[1:], columns=table[0])
                table_str = df.to_string(index=False)
                all_content.append({"page_number": page_num, "type": f"table_{table_num}", "content": table_str})

    print(f"✅ Successfully processed {len(pdf.pages)} pages. Found {len(all_content)} content blocks.")
    return all_content

# ==============================================================================
# PART 2: CHUNKING PROCESSOR
# ==============================================================================

def get_embedding_model_for_chunking() -> AzureOpenAIEmbeddings:
    """Initializes Azure embeddings specifically for the chunking process."""
    return AzureOpenAIEmbeddings(
    model=EMBEDDINGS_MODEL_NAME,
    azure_endpoint=AZURE_EMBEDDING_ENDPOINT,
    api_key=AZURE_EMBEDDING_API,
    openai_api_version='2023-05-15'
)

def chunk_content(extracted_content: List[Dict[str, Any]]) -> List[Document]:
    """Chunks content using a two-step recursive and semantic process."""
    print("-" * 50)
    print("🧠 Starting the chunking process...")
    
    embeddings_model = get_embedding_model_for_chunking()
    
    recursive_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    semantic_splitter = SemanticChunker(embeddings=embeddings_model, 
    breakpoint_threshold_type="percentile")

    final_chunks = []
    
    for item in extracted_content:
        metadata = {"source_pdf": "ingested_document", "page_number": item["page_number"]}
        initial_chunks = [item["content"]]
        if item['type'] == 'text':
            initial_chunks = recursive_splitter.split_text(item["content"])
        semantic_chunks = semantic_splitter.split_text("\n---\n".join(initial_chunks))
        
        for chunk in semantic_chunks:
            final_chunks.append(Document(page_content=chunk, metadata=metadata))
    
    print(f"✅ Successfully created {len(final_chunks)} semantic chunks.")
    return final_chunks


def run_complete_pipeline() -> None:
    """Executes the full RAG pipeline from a JSON input file and returns the results."""

    begin = time.time()
    start = time.time()
    path = '''https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D'''
    is_downloaded = download_pdf(path)
    print("PDF Downloaded in: ", time.time()-start)
    # print("PDF status: ", is_downloaded)

    start = time.time()
    all_text = process_pdf()
    # print("Text: ",len(all_text),all_text[-1])
    print("Text extracted in: ", time.time()-start)
    
    start = time.time()
    all_chunks = chunk_content(all_text)
    # print("Chunks: ",len(all_chunks), all_chunks[-1])
    print("Chunks extracted in: ", time.time()-start)
    # save_docs(all_chunks,'new_chunks.json')
    print("Process ended in: ",time.time()-begin)



    
run_complete_pipeline()