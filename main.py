import os
import time
from download_pdf import download_pdf
from chunking import process_and_chunk_pdf
# from qdrant_connect import create_collection, insert_data_in_batches, delete_collection
from async_qdrant import create_collection, insert_data_parallel, delete_collection
from graph_orchestrator import parallel_orchestrator
from question_list import list_of_questions
from utility import save_responses_append
import asyncio

async def main() -> None:
    """Executes the full RAG pipeline from a JSON input file and returns the results."""

    begin = time.time()
    start = time.time()

    path = '''https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D'''

    collection_name = "my_document_store_48"

    is_downloaded = download_pdf(path)
    print("PDF status: ", is_downloaded)
    print("PDF Downloaded in: ", time.time()-start)



    start = time.time()
    all_chunks = process_and_chunk_pdf(chunk_size=500)
    print("Chunks status: ", len(all_chunks))
    print("Chunks created in: ", time.time()-start)


    start = time.time()
    is_created = await create_collection(collection_name)
    print("Collection status: ", is_created)
    print("Collection created in: ", time.time()-start)


    start = time.time()
    is_inserted = await insert_data_parallel(all_chunks,collection_name)
    print("Insert status: ", is_inserted)
    print("Chunks insert in: ", time.time()-start)

    start = time.time()
    responses = await parallel_orchestrator(list_of_questions, collection_name)
 
    print("Query status: ", len(responses))
    print("Query answered in: ", time.time()-start)

    print("Total time taken: ",time.time()-begin)

    os.remove('temp.pdf')
    print("Deleted PDF")
    
    is_deleted = await delete_collection(collection_name)
    print("Connection deleted status: ", is_deleted)

    save_responses_append(responses)

    
asyncio.run(main())