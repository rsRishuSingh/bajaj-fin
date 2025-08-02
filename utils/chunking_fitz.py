import os
import json
import re
from typing import List

import fitz  # PyMuPDF
from dotenv import load_dotenv
import time
from langchain.docstore.document import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter



load_dotenv()

def process_and_chunk_pdf(
    pdf_path: str = "temp.pdf",
    chunk_size: int = 1000,
    chunk_overlap: int = 200
) -> List[Document]:
    """
    Fast PDF extractor & chunker using fitz (PyMuPDF):
      - Bulk-extracts text from all pages and splits once.
      - Extracts tables as simple row/tab-joined strings.

    Args:
        pdf_path: Path to PDF file.
        chunk_size: Max characters per text chunk.
        chunk_overlap: Overlap between chunks.

    Returns:
        List of Document objects with metadata.
    """
    if not os.path.exists(pdf_path):
        print(f"❌ File not found: {pdf_path}")
        return []

    print(f"🗂️  Opening PDF: {pdf_path}")
    raw_text_blocks: List[str] = []
    table_docs: List[Document] = []

    # Use a 'with' statement for clean resource management
    with fitz.open(pdf_path) as doc:
        for page_num, page in enumerate(doc, start=1):
            # 1. Collect and clean text from the page
            text = page.get_text("text") or ""
            cleaned = re.sub(r"\s+", " ", text).strip()
            if cleaned:
                raw_text_blocks.append(cleaned)

            # 2. Extract tables using fitz's find_tables() method
            for tbl_idx, table in enumerate(page.find_tables(), start=1):
                # The extract() method gets the table content as a list of lists
                table_data = table.extract()
                if not table_data:
                    continue

                # Convert table data to a tab-separated string
                rows = ["\t".join(map(str, row)) for row in table_data]
                table_str = "\n".join(rows).strip()
                if not table_str:
                    continue
                
                # Create a Document for the table
                table_docs.append(Document(
                    page_content=table_str,
                    metadata={
                        "source_pdf": os.path.basename(pdf_path),
                        "page_number": page_num,
                        "type": f"table_{tbl_idx}"
                    }
                ))

    # 3. Bulk-split all collected text blocks at once
    # This logic remains identical to the original function
    if raw_text_blocks:
        print("🧠 Bulk-splitting text…")
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""], # Recommended separators
        )
        combined_text = "\n\n".join(raw_text_blocks)
        text_chunks = splitter.split_text(combined_text)

        text_docs = [
            Document(
                page_content=chunk,
                metadata={
                    "source_pdf": os.path.basename(pdf_path),
                    "type": "text",
                    "chunk_index": idx + 1
                }
            )
            for idx, chunk in enumerate(text_chunks)
        ]
    else:
        text_docs = []


    all_docs = text_docs + table_docs
    print(f"✅ Created {len(text_docs)} text chunks + {len(table_docs)} tables = {len(all_docs)} docs")
    return all_docs


def save_docs(docs: List[Document], filepath: str = "all_docs.json") -> None:
    print(f"📥  Saving {len(docs)} chunks → {filepath}")
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(
            [{"page_content": d.page_content, "metadata": d.metadata} for d in docs],
            f,
            indent=2,
            ensure_ascii=False,
        )


def load_docs(filepath: str = "all_docs.json") -> List[Document]:
    if not os.path.exists(filepath):
        return []
    print(f"📤  Loading chunks from {filepath}")
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [Document(page_content=d["page_content"], metadata=d["metadata"]) for d in data]



if __name__ == "__main__":

    docs = []
    if not docs:
        start = time.time()
        docs = process_and_chunk_pdf()
        save_docs(docs,"new_docs.json")
        print("Chunking Ended in ",time.time()-start)
