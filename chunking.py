import os
from typing import List
import re

import pdfplumber
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter


def process_and_chunk_pdf(
    pdf_path: str = "temp.pdf",
    chunk_size: int = 1000,
    chunk_overlap: int = 200
) -> List[Document]:
    """
    Fast PDF extractor & chunker:
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

    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            # collect and clean text
            text = page.extract_text() or ""
            cleaned = re.sub(r"\s+", " ", text).strip()
            if cleaned:
                raw_text_blocks.append(cleaned)

            # extract simple table strings
            for tbl_idx, table in enumerate(page.extract_tables(), start=1):
                if not table:
                    continue
                rows = ["\t".join(map(str, row)) for row in table]
                table_str = "\n".join(rows).strip()
                if not table_str:
                    continue
                table_docs.append(Document(
                    page_content=table_str,
                    metadata={
                        "source_pdf": os.path.basename(pdf_path),
                        "page_number": page_num,
                        "type": f"table_{tbl_idx}"
                    }
                ))

    # bulk-split all text blocks
    print("🧠 Bulk-splitting text…")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
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

    all_docs = text_docs + table_docs
    print(f"✅ Created {len(text_docs)} text chunks + {len(table_docs)} tables = {len(all_docs)} docs")
    return all_docs
