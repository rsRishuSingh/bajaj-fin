import time
from download_pdf import download_pdf
from chunking import process_and_chunk_pdf
from qdrant_connect import create_collection, insert_data_in_batches
from graph_orchestrator import graph_orchestrator_run
from question_list import list_of_questions
from utility import save_responses

def main() -> None:
    """Executes the full RAG pipeline from a JSON input file and returns the results."""

    begin = time.time()
    start = time.time()

    path = '''https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D'''
    
    is_downloaded = download_pdf(path)
    print("PDF status: ", is_downloaded)
    print("PDF Downloaded in: ", time.time()-start)



    start = time.time()
    all_chunks = process_and_chunk_pdf()
    print("Chunks status: ", len(all_chunks))
    print("Chunks created in: ", time.time()-start)


    collection_name = "my_document_store_45"

    start = time.time()
    is_created = create_collection(collection_name)
    print("Collection status: ", is_created)
    print("Collection created in: ", time.time()-start)


    start = time.time()
    is_inserted = insert_data_in_batches(all_chunks,collection_name)
    print("Insert status: ", is_inserted)
    print("Chunks insert in: ", time.time()-start)

    start = time.time()
    responses = graph_orchestrator_run(list_of_questions, collection_name)
    save_responses(responses)
    print("Query status: ", len(responses))
    print("Query answered in: ", time.time()-start)

    print("Total time taken: ",time.time()-begin)

    
main()