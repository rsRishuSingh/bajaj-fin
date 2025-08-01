import os
import json
import re
from dotenv import load_dotenv
from typing import List

import fitz  # PyMuPDF for PDF parsing
from langchain.docstore.document import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker

from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever

load_dotenv()
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
GROQ_MODEL_NAME = "qwen/qwen3-32b"
COLLECTION_NAME = "TESLA_RAG_DOCS"
CHROMA_DB_PATH = "chromaDB/saved/"
PDF_DIR = "PDFs/"
PDF_FILES = ["TESLA"]
ALL_DOCS_JSON = "all_docs.json"
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USERNAME")
NEO4J_PW = os.getenv("NEO4J_PASSWORD")
NEO4J_INDEX = os.getenv("NEO4J_INDEX", "LangChainDocs")


def recursive_split(text: str, chunk_size: int = 500, chunk_overlap: int = 100) -> List[str]:
    """
    Split text into character-based chunks with specified size and overlap.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " "]
    )
    return splitter.split_text(text)


def semantic_chunker(text: str, embeddings_model) -> List[str]:
    """
    Further break text into semantically coherent chunks using embeddings_model.
    """
    chunks: List[str] = []
    for segment in recursive_split(text):
        chunker = SemanticChunker(embeddings_model)
        chunks.extend(chunker.split_text(segment))
    return chunks


def extract_chunks_from_pdf(pdf_path: str, embeddings_model) -> List[Document]:
    """
    Parse PDF at pdf_path, split each page's text into semantic chunks, and wrap in Document objects.
    """
    documents: List[Document] = []
    print('🗂️  Getting PDF...\n\n')
    pdf = fitz.open(pdf_path)
    for page_idx, page in enumerate(pdf):
        print('📖 Reading Page no: ', page_idx+1)
        raw = re.sub(r"\s+", " ", page.get_text("text")).strip()
        if not raw:
            continue
        for idx, chunk in enumerate(semantic_chunker(raw, embeddings_model)):
            documents.append(
                Document(
                    page_content=chunk,
                    metadata={"page": page_idx + 1, "chunk": idx, "source": os.path.basename(pdf_path)}
                )
            )
    pdf.close()
    return documents


def save_docs(docs: List[Document], filepath: str = ALL_DOCS_JSON) -> None:
    """
    Serialize list of Documents to JSON for later reuse.
    """
    print("📥📄 Saving chunks for future use ")
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(
            [{"page_content": d.page_content, "metadata": d.metadata} for d in docs],
            f,
            indent=2
        )


def load_docs(filepath: str = ALL_DOCS_JSON) -> List[Document]:
    """
    Load Document list from JSON if exists, else return empty list.
    """
    print("📤📄 Loading chunks ")
    if not os.path.exists(filepath):
        return []
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [Document(page_content=item["page_content"], metadata=item["metadata"]) for item in data]


def init_chroma() -> Chroma:
    """
    Initialize or load a Chroma collection with local HuggingFace embeddings.
    """
    print('🧭 Creating ChromaDB...')
    return Chroma(
        persist_directory=CHROMA_DB_PATH,
        collection_name=COLLECTION_NAME,
        embedding_function=HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME)
    )


def main() -> None:
    """
    Full RAG pipeline: extract/load docs, initialize vector stores, build ensemble retriever, and serve QA.
    """
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME)
    docs = load_docs()
    if not docs:
        for name in PDF_FILES:
            path = os.path.join(PDF_DIR, f"{name}.pdf")
            if os.path.exists(path):
                docs.extend(extract_chunks_from_pdf(path, embeddings))
        save_docs(docs)

    chroma_store = init_chroma()

    if chroma_store._collection.count() == 0:
        chroma_store.add_documents(docs)

    chroma_retriever = chroma_store.as_retriever(
        search_type="mmr",  # avoids returning documents that are too similar to each other
        search_kwargs={"k": 5, "fetch_k": 10}
    )
    bm25_retriever = BM25Retriever.from_texts(
        [d.page_content for d in docs],
        metadatas=[d.metadata for d in docs],
        k=5
    )
    
    ensemble = EnsembleRetriever(
        retrievers=[bm25_retriever, chroma_retriever],
        weights=[1, 1]
    )

    print("--- RAG System Ready ---")
    while True:
        query = input('❓ What do you want to know: ')
        if query.lower() == 'quit':
            break
        docs_out = ensemble.invoke(query)
        context = "\n\n".join(doc.page_content for doc in docs_out)
        messages = [
            {"role": "system", "content": "You are an expert assistant. Use only provided context."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion:\n{query}"}
        ]
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        resp = client.chat.completions.create(
            model=os.getenv("GROQ_MODEL_NAME", GROQ_MODEL_NAME),
            messages=messages,
            temperature=0.2
        )
        print(resp.choices[0].message.content)


if __name__ == "__main__":
    main()
