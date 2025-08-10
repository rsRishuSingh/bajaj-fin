import fitz  # PyMuPDF
import json
import numpy as np
from typing import List, Dict, Tuple
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi
from langchain_core.messages import SystemMessage

# UTILITY FUNCTIONS 

def process_page_text(page_data: Tuple[bytes, int]) -> Tuple[str, int]:
    """
    Extracts text from a single page of a PDF's content using PyMuPDF.
    This version avoids rendering the page, making it more robust.
    """
    pdf_content, page_num_0_indexed = page_data
    text = ""
    page_num_1_indexed = page_num_0_indexed + 1
    try:
        with fitz.open(stream=pdf_content, filetype="pdf") as doc:
            page = doc.load_page(page_num_0_indexed)
            text = page.get_text("text", sort=True) or ""
    except Exception as e:
        print(f"Error processing page {page_num_1_indexed}: {e}")
    return text, page_num_1_indexed

def chunk_text_with_metadata(full_text: str, pages_text: List[Dict]) -> List[Dict]:
    """Performs text splitting and attaches page number metadata."""
    print("🔪 Performing semantic chunking in a separate process...")
    semantic_splitter = RecursiveCharacterTextSplitter(chunk_size=768, chunk_overlap=128)
    documents = semantic_splitter.create_documents([full_text])

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
    """Initializes the BM25 retriever in a separate process."""
    print("🔍 Initializing BM25 retriever in a separate process...")
    tokenized_corpus = [doc.lower().split() for doc in chunk_texts]
    return BM25Okapi(tokenized_corpus)

def get_context(state: dict, max_messages: int = 20) -> str:
    """Formats the conversation history from the agent state into a readable string."""
    formatted_history = ""
    messages = state.get('messages', [])
    if not messages: return "No history available."
    for msg in messages[-max_messages:]:
        if isinstance(msg, SystemMessage): continue
        role = "Human" if msg.type == "human" else "Assistant"
        content = msg.content
        tool_calls = msg.tool_calls if hasattr(msg, "tool_calls") and msg.tool_calls else []
        if tool_calls:
            tool_info = [f"Tool: {tc['name']}, Args: {json.dumps(tc['args'])}" for tc in tool_calls]
            content += f"\n[Tool Calls: {'; '.join(tool_info)}]"
        formatted_history += f"{role}: {content}\n"
    return formatted_history