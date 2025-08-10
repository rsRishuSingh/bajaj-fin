import asyncio
import time
import json
from typing import List
from log import log_and_save_response

from langchain_core.messages import HumanMessage

from .config import QDRANT_URL, QDRANT_API_KEY
from .rag_system import MultiDocumentRAG
from .orchestrator import Orchestrator


# MAIN EXECUTION LOGIC

async def run_pipeline(doc_url: str, questions: List[str]):
    """
    Encapsulates the entire document processing and Q&A pipeline.
    """
    total_start_time = time.time()

    # --- System Initialization ---
    rag_system = MultiDocumentRAG(QDRANT_URL, QDRANT_API_KEY)
    await rag_system.prepare_document(doc_url)
    doc_processing_end_time = time.time()

    if rag_system.ingestion_error_message:
        print(rag_system.ingestion_error_message)
        return []

    # --- HYBRID STRATEGY: Initial Document-Level Routing ---
    print("\n" + "=" * 80)
    print("🚦 Performing initial document analysis for execution strategy...")

    is_complex_strategy = False
    is_static_file = any(
        doc_url.lower().split('?')[0].endswith(ext)
        for ext in ['.pdf', '.docx', '.pptx', '.xlsx', '.zip', '.txt']
    )

    if not is_static_file:
        is_complex_strategy = True
        print(
            "   - ✅ Strategy: Non-Static URL detected. Executing questions SEQUENTIALLY via Agent.")
    else:
        full_document_text = "\n".join([c.get('text', '') for c in rag_system.active_chunks])
        num_chunks = len(rag_system.active_chunks)

        if num_chunks > 0:
            lower_text = full_document_text.lower()
            keyword_count = lower_text.count("api") + lower_text.count("http")
            keyword_density = keyword_count / num_chunks

            if keyword_density > 0.25:
                is_complex_strategy = True
                print(
                    f"   - ✅ Strategy: High Keyword Density ({keyword_density:.2f}). Executing questions SEQUENTIALLY via Agent.")
            else:
                print(
                    f"   - ✅ Strategy: Low Keyword Density ({keyword_density:.2f}). Executing all questions in PARALLEL via RAG.")
        else:
            print(f"   - ✅ Strategy: No content found. Defaulting to PARALLEL via RAG.")

    print("=" * 80)

    qna_results = []
    q_start_time = time.time()

    if not is_complex_strategy:
        # --- Execution Path 1: Parallel RAG ---
        answer_tasks = [rag_system.answer_question(q) for q in questions]
        final_answers = await asyncio.gather(*answer_tasks)

        for q, a in zip(questions, final_answers):
            print("\n" + "#" * 80)
            print(f"❓ Query: {q}")
            print(f"✅ Answer: {a}")
            qna_results.append({"question": q, "answer": a})
    else:
        # --- Execution Path 2: Sequential Agent ---
        orchestrator = Orchestrator(rag_system)

        for q in questions:
            print("\n" + "#" * 80)
            print(f"❓ Query: {q}")
            print("#" * 80)

            initial_state = {
                "query": q,
                "chunks": rag_system.active_chunks,
                "messages": [HumanMessage(content=q)]
            }
            final_state = await orchestrator.app.ainvoke(initial_state, {"recursion_limit": 30})

            final_answer = "Could not determine a final answer."
            if final_state and final_state.get('messages'):
                final_answer = final_state['messages'][-1].content

            print("\n" + "=" * 80)
            print(f"✅ Final Answer: {final_answer}")
            print("=" * 80)
            qna_results.append({"question": q, "answer": final_answer})

    # --- Final Reporting ---
    total_q_time = time.time() - q_start_time
    total_execution_time = time.time() - total_start_time
    doc_processing_time = doc_processing_end_time - total_start_time

    print("\n\n" + "=" * 80)
    print("💾 Saving detailed Q&A results to a JSON file...")
    output_data = {
        "document_url": doc_url,
        "questions_and_answers": qna_results,
        "timing_stats": {
            "document_processing_loading_seconds": round(doc_processing_time, 2),
            "total_querying_seconds": round(total_q_time, 2),
            "average_time_per_question": round(total_q_time / len(questions),
                                               2) if questions else 0,
            "total_execution_seconds": round(total_execution_time, 2)
        }
    }
    savefileName = log_and_save_response(output_data, True)

    print(f"✅ Results successfully saved to: {savefileName}")

    print("\n" + "=" * 80)
    print("📊 FINAL TIMING REPORT")
    print("=" * 80)
    print(f"⏱️ Document Processing/Loading Time: {doc_processing_time:.2f} seconds.")
    print(f"⏱️ Total Questioning Time: {total_q_time:.2f} seconds for {len(questions)} questions.")
    if questions:
        print(f"⏱️ Average Time Per Question: {total_q_time / len(questions):.2f} seconds.")
    print(f"⏱️ Total Execution Time: {total_execution_time:.2f} seconds.")
    print("=" * 80)

    # Return a list of answers in order
    return [item["answer"] for item in qna_results]