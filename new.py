import os
import json
import uuid
import requests
from typing import List, Dict, Any, TypedDict

# --- Library Imports ---
from dotenv import load_dotenv
import pandas as pd
import pdfplumber
from qdrant_client import QdrantClient, models
from sentence_transformers import CrossEncoder
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_openai import AzureOpenAIEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langgraph.graph import StateGraph, END

# --- Global Configuration ---
load_dotenv()
COLLECTION_NAME = "my_final_policy_documents"
AZURE_ADA_002_VECTOR_SIZE = 1536
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
EMBEDDINGS_MODEL_NAME = "text-embedding-ada-002"
AZURE_EMBEDDING_ENDPOINT = os.getenv('azure_embedding_endpoint')
AZURE_EMBEDDING_API = os.getenv('azure_embedding_api')

# ==============================================================================
# PART 1: DOCUMENT PROCESSOR
# ==============================================================================

def process_pdf(pdf_path: str) -> List[Dict[str, Any]]:
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
        
        initial_chunks = recursive_splitter.split_text(item["content"])
        semantic_chunks = semantic_splitter.split_text("\n---\n".join(initial_chunks))
        
        for chunk in semantic_chunks:
            final_chunks.append(Document(page_content=chunk, metadata=metadata))
    
    print(f"✅ Successfully created {len(final_chunks)} semantic chunks.")
    return final_chunks

# ==============================================================================
# PART 3: VECTOR STORE PROCESSOR
# ==============================================================================

def store_in_qdrant(documents: List[Document]):
    """Embeds documents and stores them in Qdrant, enabling BM25 search."""

    qdrant_client = QdrantClient(url=os.getenv("QDRANT_URL"), api_key=os.getenv("QDRANT_API_KEY"))
    dense_model = get_embedding_model_for_chunking()
    
    if not qdrant_client.collection_exists(collection_name=COLLECTION_NAME):
        qdrant_client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(size=AZURE_ADA_002_VECTOR_SIZE, distance=models.Distance.COSINE),
        )
        qdrant_client.create_payload_index(
            collection_name=COLLECTION_NAME, field_name="text",
            field_schema=models.TextIndexParams(type="text", tokenizer=models.TokenizerType.WORD, lowercase=True)
        )
    else:
        print(f"Collection '{COLLECTION_NAME}' already exists.")
        
    batch_size = 128
    for i in range(0, len(documents), batch_size):
        batch_docs = documents[i : i + batch_size]
        batch_content = [doc.page_content for doc in batch_docs]
        embeddings = dense_model.embed_documents(batch_content)
        points = [
            models.PointStruct(id=str(uuid.uuid4()), vector=embedding, payload={**doc.metadata, "text": doc.page_content})
            for doc, embedding in zip(batch_docs, embeddings)
        ]
        qdrant_client.upsert(collection_name=COLLECTION_NAME, points=points, wait=True)
    print(f"✅ Successfully stored all {len(documents)} documents.")



# ==============================================================================
# PART 5: MAIN APPLICATION RUNNER
# ==============================================================================

def download_pdf(url: str, save_path: str) -> bool:
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

def run_complete_pipeline(json_path: str) -> list:
    """Executes the full RAG pipeline from a JSON input file and returns the results."""

    with open(json_path, 'r') as f: input_data = json.load(f)
    pdf_url, questions = input_data["pdf_blob_url"], input_data["list_of_questions"]
    local_pdf_path = "temp_document.pdf"

    print("### STARTING INGESTION PIPELINE ###")
    if not download_pdf(pdf_url, local_pdf_path): return []
    extracted_content = process_pdf(local_pdf_path)
    if not extracted_content: return []
    documents = chunk_content(extracted_content)
    if not documents: return []
    store_in_qdrant(documents)
    print("### INGESTION PIPELINE COMPLETE ###")
    
    agent = build_agent()
    final_results = []
    for i, question in enumerate(questions):
        print("\n" + "="*60)
        print(f"### PROCESSING QUESTION {i+1}/{len(questions)}: '{question}' ###")
        result = agent.invoke({"original_question": question})
        answer = result.get('answer', 'No answer was generated.')
        final_results.append({"question": question, "answer": answer})
        print(f"### ANSWER GENERATED ###\n{answer}")
    
    os.remove(local_pdf_path)
    print(f"\n✅ Pipeline complete. Cleaned up temporary file.")
    return final_results

if __name__ == '__main__':
    INPUT_JSON_PATH = r"D:\2. hackrx\hackrx\input_user.json"
    if not os.path.exists(INPUT_JSON_PATH):
        print(f"❌ Error: Input file not found at '{INPUT_JSON_PATH}'")
    else:
        list_of_answers = run_complete_pipeline(INPUT_JSON_PATH)
        print("\n" + "#"*60)
        print("## FINAL STRUCTURED OUTPUT ##")
        print("#"*60)
        print(json.dumps(list_of_answers, indent=2))