import os
import json
import re
from typing import List

import fitz  # PyMuPDF
from dotenv import load_dotenv

from langchain.docstore.document import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings


load_dotenv()

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
embeddings_model = HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME)


def recursive_split(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " "],
    )
    return splitter.split_text(text)


def semantic_chunker(text: str) -> List[str]:
    """
    Given a block of text, first apply recursive_split(), then
    semantically re‑split each segment via Azure embeddings.
    """
    chunks: List[str] = []
    for segment in recursive_split(text):
        chunker = SemanticChunker(embeddings_model)
        chunks.extend(chunker.split_text(segment))
    return chunks

def extract_chunks_from_pdf(pdf_path: str) -> List[Document]:
    docs: List[Document] = []
    print(f"🗂️  Opening PDF: {pdf_path}")
    pdf = fitz.open(pdf_path)

    for page_idx, page in enumerate(pdf, start=1):
        print(f"📖  Reading page {page_idx}")
        raw = re.sub(r"\s+", " ", page.get_text("text")).strip()
        if not raw:
            continue

        for chunk_idx, chunk in enumerate(recursive_split(raw)):
            docs.append(
                Document(
                    page_content=chunk,
                    metadata={
                        "page": page_idx,
                        "chunk": chunk_idx,
                        "source": os.path.basename(pdf_path),
                    },
                )
            )
    pdf.close()
    return docs


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

    PDF_PATH = "Policy.pdf"
    # Avoid re‑chunking if you’ve already saved:
    docs = []
    if not docs:
        docs = extract_chunks_from_pdf(PDF_PATH)
        save_docs(docs,"new_docs.json")

    print(f"✅ Ready with {len(docs)} semantic chunks.")
