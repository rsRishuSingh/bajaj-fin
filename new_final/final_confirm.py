import os
import fitz  # PyMuPDF
import json
import time
import numpy as np
import asyncio
import aiohttp
import pickle
import hashlib
from concurrent.futures import ProcessPoolExecutor
from typing import List, Dict, Tuple, Optional

from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from qdrant_client import QdrantClient, models, AsyncQdrantClient
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi

# --- HELPER FUNCTIONS FOR PARALLEL CPU-BOUND WORK ---

#config
load_dotenv()
azure_openai_chat_endpoint = os.getenv("AZURE_OPENAI_CHAT_ENDPOINT")
azure_openai_chat_api = os.getenv("AZURE_OPENAI_CHAT_API_KEY")
azure_openai_emb_endpoint=os.getenv("AZURE_OPENAI_EMB_ENDPOINT")
azure_openai_emb_api=os.getenv("AZURE_OPENAI_EMB_API_KEY")
qdrant_url = os.getenv("QDRANT_URL")
qdrant_api_key = os.getenv("QDRANT_API_KEY")


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
            # Directly load the page and extract text without re-rendering
            page = doc.load_page(page_num_0_indexed)
            text = page.get_text("text", sort=True) or ""
    except Exception as e:
        print(f"Error processing page {page_num_1_indexed}: {e}")
    return text, page_num_1_indexed

def chunk_text_with_metadata(full_text: str, pages_text: List[Dict]) -> List[Dict]:
    """Performs text splitting and attaches page number metadata."""
    print("🔪 Performing semantic chunking in a separate process...")
    semantic_splitter = RecursiveCharacterTextSplitter(
        chunk_size=768, chunk_overlap=128
    )
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


class MultiDocumentRAG:
    """
    A production-ready RAG system that can manage and query multiple documents,
    processing each document only once and storing it permanently.
    """

    def __init__(self, qdrant_url: str, qdrant_api_key: str):
        # --- USER CONFIGURATION ---
        self.llm = AzureChatOpenAI(
            deployment_name="gpt-4.1-mini",
            azure_endpoint=azure_openai_chat_endpoint,
            api_version="2024-12-01-preview",
            api_key=azure_openai_chat_api,
            temperature=0, request_timeout=60
        )
        self.embedder = AzureOpenAIEmbeddings(
                azure_deployment="text-embedding-ada-002",
                openai_api_version="2023-05-15",
                azure_endpoint=azure_openai_emb_endpoint,
                api_key=azure_openai_emb_api,
)
        
        print("☁️ Connecting to Qdrant Cloud...")
        self.sync_qdrant_client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key, timeout=60)
        self.qdrant_client = AsyncQdrantClient(url=qdrant_url, api_key=qdrant_api_key, timeout=60)
        
        self.data_dir = "rag_data_library"
        os.makedirs(self.data_dir, exist_ok=True)
        
        self.active_collection_name: Optional[str] = None
        self.active_bm25: Optional[BM25Okapi] = None
        self.active_chunks: List[Dict] = []

    def _get_collection_name_from_content(self, pdf_content: bytes) -> str:
        """Creates a unique, safe collection name from the PDF's content hash."""
        content_hash = hashlib.sha256(pdf_content).hexdigest()
        return f"doc_content_{content_hash[:16]}"

    async def prepare_document(self, pdf_url: str, force_reingest: bool = False):
        """
        Ensures a document is ready for querying. It will either load existing data
        or trigger a full ingestion if the document is new.
        """
        print("\n" + "="*80)
        print(f"🚀 Preparing document from URL: {pdf_url}")

        print(f"   - 📥 Downloading PDF to verify identity...")
        async with aiohttp.ClientSession() as session:
            async with session.get(pdf_url, timeout=120) as response:
                response.raise_for_status()
                pdf_content = await response.read()

        collection_name = self._get_collection_name_from_content(pdf_content)
        print(f"   - Mapped to Content-Based Collection: {collection_name}")

        if self.active_collection_name == collection_name and not force_reingest:
            print("   - ✅ Document is already active. Ready to query.")
            print("="*80)
            return

        bm25_path = os.path.join(self.data_dir, f"{collection_name}_bm25.pkl")
        chunks_path = os.path.join(self.data_dir, f"{collection_name}_chunks.json")

        if os.path.exists(bm25_path) and os.path.exists(chunks_path) and not force_reingest:
            print("   -  Found existing local data. Loading from disk...")
            with open(bm25_path, 'rb') as f_bm25:
                self.active_bm25 = pickle.load(f_bm25)
            with open(chunks_path, 'r', encoding='utf-8') as f_chunks:
                self.active_chunks = json.load(f_chunks)
            self.active_collection_name = collection_name
            print("   - ✅ Document loaded. Ready to query.")
            print("="*80)
            return

        print("   - No existing data found. Starting full ingestion process...")
        await self._ingest_new_document(pdf_url, pdf_content, collection_name, bm25_path, chunks_path)
        print("="*80)

    async def _ingest_new_document(self, pdf_url: str, pdf_content: bytes, collection_name: str, bm25_path: str, chunks_path: str):
        """The internal pipeline for processing and storing a new document."""
        try:
            all_collections = self.sync_qdrant_client.get_collections().collections
            collection_names = [collection.name for collection in all_collections]

            if collection_name not in collection_names:
                print(f"   - Collection '{collection_name}' not found. Creating it now...")
                embedding_size = len(self.embedder.embed_query("test"))
                self.sync_qdrant_client.create_collection(
                    collection_name=collection_name,
                    vectors_config=models.VectorParams(size=embedding_size, distance=models.Distance.COSINE),
                )
                print("   - ✅ Collection created successfully.")
            else:
                print(f"   - ✅ Collection '{collection_name}' already exists.")
        except Exception as e:
            print(f"   - 🚨 CRITICAL ERROR during Qdrant collection check/creation: {e}")
            return

        process_pool = ProcessPoolExecutor()
        loop = asyncio.get_running_loop()
        try:
            print("   - 📖 Extracting text in parallel...")
            with fitz.open(stream=pdf_content, filetype="pdf") as doc:
                # FIX: Pass the full pdf_content and 0-indexed page number to each worker
                # This avoids the error-prone doc.convert_to_pdf() rendering step.
                tasks = [loop.run_in_executor(process_pool, process_page_text, (pdf_content, p)) for p in range(len(doc))]
                page_results = await asyncio.gather(*tasks)
            
            page_results.sort(key=lambda x: x[1])
            pages_text = [{"page": page_num, "text": text} for text, page_num in page_results]
            full_text = "\n\n".join([p['text'] for p in pages_text])

            chunks = await loop.run_in_executor(process_pool, chunk_text_with_metadata, full_text, pages_text)
            chunk_texts = [c['text'] for c in chunks]
            bm25 = await loop.run_in_executor(process_pool, initialize_bm25, chunk_texts)
            
            print(f"   - ✅ Created {len(chunks)} chunks.")

            print("   - 💾 Saving BM25 model and chunks to local disk...")
            with open(bm25_path, 'wb') as f_bm25:
                pickle.dump(bm25, f_bm25)
            with open(chunks_path, 'w', encoding='utf-8') as f_chunks:
                json.dump(chunks, f_chunks, indent=2)

            print("   - 🧠 Embedding and uploading chunks to Qdrant...")
            batch_size = 64
            for i in range(0, len(chunks), batch_size):
                batch_chunks = chunks[i:i+batch_size]
                batch_contents = [c['text'] for c in batch_chunks]
                batch_embeddings = await self.embedder.aembed_documents(batch_contents)
                await self.qdrant_client.upsert(
                    collection_name=collection_name,
                    points=models.Batch(
                        ids=list(range(i, i + len(batch_chunks))),
                        vectors=batch_embeddings,
                        payloads=[{"text": c['text'], "page": c['page'], "source_url": pdf_url} for c in batch_chunks]
                    ),
                    wait=True
                )
            
            self.active_collection_name = collection_name
            self.active_bm25 = bm25
            self.active_chunks = chunks
            print("   - ✅ Ingestion complete. Document is now active.")
        finally:
            process_pool.shutdown()

    async def answer_question(self, question: str) -> str:
        """Answers a question using the currently active document."""
        if not self.active_bm25 or not self.active_chunks or not self.active_collection_name:
            return "Error: No document is active. Please call `prepare_document(pdf_url)` first."
        
        prompt_expand = f"""You are an expert document analyst. Your task is to deconstruct a user's conversational or vague question into a set of clear, specific, and standalone queries that can be used to search and extract relevant information from any structured or unstructured document. These queries should comprehensively cover all aspects of the user's original intent, such as definitions, scope, limitations, procedures, timelines, exclusions, and conditions—depending on the context of the question and nature of the document.

--- EXAMPLES ---

Original Question: I did a cataract treatment of Rs 100,000. Will you settle the full Rs 100,000?
Generated Queries:
Is cataract treatment a covered medical procedure?
What is the maximum limit or sub-limit payable for cataract surgery?
Are there specific conditions or waiting periods applicable to cataract treatment?
What are the policy's general exclusions related to eye surgeries or treatments?

Original Question: When will my root canal claim of Rs 25,000 be settled?
Generated Queries:
Is root canal treatment covered under the policy's dental benefits?
What is the process and typical timeline for claim settlement?
Are there monetary limits or sub-limits for outpatient dental procedures?
What are the waiting periods associated with dental treatments?

Original Question: Will this software support live video editing?
Generated Queries:
Does the software have live video editing capabilities?
Are there any performance or hardware requirements for live video editing?
Which file formats are supported for live editing?
Is there a limit to video resolution or length for real-time editing?

Original Question: Can I use this policy while traveling outside India?
Generated Queries:
Is international coverage included in this policy?
Are there any geographical exclusions or limitations?
What are the procedures for filing a claim from outside India?
Are there additional charges or riders for international usage?

--- YOUR TASK ---

Original Question: {question}
Generated Queries:
"""

        try:
            response = await self.llm.ainvoke(prompt_expand)
            expanded_queries = response.content.strip().split("\n")
            all_queries = [question] + [q.strip() for q in expanded_queries if q.strip()]
        except Exception:
            all_queries = [question]
        
        fused_scores = {}
        k = 60
        for q in all_queries:
            tokenized_query = q.lower().split()
            bm25_scores = self.active_bm25.get_scores(tokenized_query)
            bm25_top_indices = np.argsort(bm25_scores)[::-1][:15]
            
            for i, doc_id in enumerate(bm25_top_indices):
                fused_scores[doc_id] = fused_scores.get(doc_id, 0) + 1 / (k + i + 1)

            query_embedding = await self.embedder.aembed_query(q)
            semantic_results = await self.qdrant_client.search(
                collection_name=self.active_collection_name, query_vector=query_embedding, limit=15
            )
            for i, hit in enumerate(semantic_results):
                fused_scores[hit.id] = fused_scores.get(hit.id, 0) + 1 / (k + i + 1)
        
        sorted_unique_ids = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
        retrieved_chunks = [self.active_chunks[doc_id] for doc_id, score in sorted_unique_ids[:8] if doc_id < len(self.active_chunks)]
        
        context = "\n\n---\n\n".join([f"[CONTEXT {i+1} | Source: Page {c.get('page', 'N/A')}]\n{c['text']}" for i, c in enumerate(retrieved_chunks)])
        prompt_generate = f"""You are a meticulous and diligent insurance policy analyst. Your primary goal is to answer the user's question with unwavering accuracy, using ONLY the provided text snippets from a policy document.

**CRITICAL INSTRUCTIONS:**
1.  **Base all answers strictly on the provided CONTEXT.** Do not use any outside knowledge.
2.  **Analyze the entire context.** The answer may require combining information from multiple context snippets.
3.  **Give a direct, precise, and concise answer.** Avoid long answers, if possible based on the question.
4.  **When answering the query, if the response is based on a specific clause or section in the document, please explicitly mention the clause number or title (e.g., 'as per Clause 4.2' or 'Section II – Coverage Details'). This ensures traceability of the answer to the exact source in the document.
5.  **Format the final answer as a single, unbroken paragraph of text.** Do not use any line breaks, bullet points, or paragraph separations. The entire response must be one continuous block of plain text.

**HOW TO HANDLE QUESTIONS ABOUT COVERAGE:**
Your default is not to say "I don't know". You must follow this specific logic:
- **STEP 1: LOOK FOR DIRECT CONFIRMATION:** Does the context explicitly state the item IS covered? If yes, state that and explain the details (e.g., limits, amounts).
- **STEP 2: LOOK FOR DIRECT EXCLUSIONS:** If you can't find direct confirmation, actively search for exclusion clauses. If the context says an item is 'not covered', 'excluded', or 'not payable', your answer MUST be "No, it is not covered." and you MUST state the reason based on the exclusion clause.
- **STEP 3: LOOK FOR UNMET CONDITIONS:** If it's not explicitly excluded, check for conditions (e.g., waiting periods, age limits, BMI requirements, required duration). If the user's question fails to meet a condition, your answer MUST be "No, it is not covered under these circumstances because..." and explain the condition that isn't met.
- **STEP 4: LAST RESORT:** Only if you have exhausted all the steps above and find no relevant information (neither confirming, excluding, nor conditional), should you state: "Based on the provided text, there is no specific information regarding this." Do not use this as an easy way out.

---
**CONTEXT:**
{context}
---
**QUESTION:** {question}
---
**ANALYST'S ANSWER:**"""
        response = await self.llm.ainvoke(prompt_generate)
        return response.content.strip()

async def main():
    """Main function to demonstrate the production-ready workflow with parallel questioning."""
    # --- Start total timer ---
    total_start_time = time.time()

    # --- USER CONFIGURATION ---
    QDRANT_URL = qdrant_url
    QDRANT_API_KEY = qdrant_api_key
    PDF_URL = "https://hackrx.blob.core.windows.net/assets/principia_newton.pdf?sv=2023-01-03&st=2025-07-28T07%3A20%3A32Z&se=2026-07-29T07%3A20%3A00Z&sr=b&sp=r&sig=V5I1QYyigoxeUMbnUKsdEaST99F5%2FDfo7wpKg9XXF5w%3D"
    QUESTIONS = [
  "What are the key arguments made in the dedication to the teachers of the Normal School of New York?",
  "Why did Newton prefer the geometrical method over algebraic approaches, according to the American introduction?",
  "What was Newton's first invention as a child that demonstrated his mechanical skills?",
  "How did Newton's early education at Grantham influence his scientific curiosity?",
  "What were the reasons Newton rejected refracting telescopes in favor of reflecting telescopes?",
  "How did Newton describe his discovery of the binomial theorem and fluxions?",
  "What optical discoveries did Newton make using a prism and what conclusions did he draw?",
  "Why did Newton abandon his early calculation of the moon's orbit under gravity?",
  "How did Newton’s method of fluxions differ from the earlier work of Wallis and others?",
  "What advice did Newton give to his friend Francis Aston about traveling and observation?"
]
    
    # Initialize the RAG system
    rag_system = MultiDocumentRAG(qdrant_url=QDRANT_URL, qdrant_api_key=QDRANT_API_KEY)

    # --- Step 1: Prepare the document. It will ingest only if new. ---
    await rag_system.prepare_document(PDF_URL)
    pdf_processing_end_time = time.time()
    
    # --- Step 2: Ask all questions IN PARALLEL against the prepared document ---
    print("\n\n🎯 Querying Document in Parallel...")
    
    answer_tasks = [rag_system.answer_question(q) for q in QUESTIONS]
    q_start_time = time.time()
    final_answers = await asyncio.gather(*answer_tasks)
    total_q_time = time.time() - q_start_time

    # --- Step 3: Display results in the console ---
    for question, answer in zip(QUESTIONS, final_answers):
        print("\n" + "=" * 60)
        print(f"📝 Question: {question}")
        print(f"✅ Answer: {answer}")

    # --- Step 4: Calculate final timing stats ---
    total_execution_time = time.time() - total_start_time
    pdf_processing_time = pdf_processing_end_time - total_start_time

    # --- Step 5: Save detailed results to a JSON file ---
    print("\n\n" + "="*80)
    print("💾 Saving detailed Q&A results to a JSON file...")
    output_data = {
        "document_url": PDF_URL,
        "questions_and_answers": [
            {"question": q, "answer": a} for q, a in zip(QUESTIONS, final_answers)
        ],
        "timing_stats": {
            "pdf_processing_loading_seconds": round(pdf_processing_time, 2),
            "total_parallel_querying_seconds": round(total_q_time, 2),
            "average_time_per_question": round(total_q_time / len(QUESTIONS), 2) if QUESTIONS else 0,
            "total_execution_seconds": round(total_execution_time, 2)
        }
    }
    
    output_filename = "rag_session_results.json"
    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=4, ensure_ascii=False)
    
    print(f"✅ Results successfully saved to: {output_filename}")


    # --- Step 6: Final Timing Report in Console ---
    print("\n" + "="*80)
    print("📊 FINAL TIMING REPORT")
    print("="*80)
    print(f"⏱️ PDF Processing/Loading Time: {pdf_processing_time:.2f} seconds.")
    print(f"⏱️ Total Parallel Questioning Time: {total_q_time:.2f} seconds for {len(QUESTIONS)} questions.")
    print(f"⏱️ Average Time Per Question (in parallel): {total_q_time / len(QUESTIONS):.2f} seconds.")
    print(f"⏱️ Total Execution Time: {total_execution_time:.2f} seconds.")
    print("="*80)


if __name__ == "__main__":
    asyncio.run(main())