import sys
import os
import json
import time

import asyncio
from typing import List, Annotated, Sequence, TypedDict, Dict, Tuple
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
# Use the updated AzureChatOpenAI from langchain-openai (supports bind_tools)
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages
from langchain.docstore.document import Document
from utility import append_to_response, get_context
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")




# Define recursion limit for deep workflows (support 20 queries in parallel)
sys.setrecursionlimit(10**5)

load_dotenv()

# Initialize LLM model
# Azure OpenAI configuration
AZURE_ENDPOINT        = "https://rishu-mdodjz43-eastus2.cognitiveservices.azure.com/"
AZURE_DEPLOYMENT_NAME = "gpt-4o"
AZURE_API_VERSION     = "2024-05-01-preview"
AZURE_API_KEY         = os.getenv("AZURE_OPENAI_GPT_API")

llm = AzureChatOpenAI(
    deployment_name=AZURE_DEPLOYMENT_NAME,
    azure_endpoint=AZURE_ENDPOINT,
    api_version=AZURE_API_VERSION,
    api_key=AZURE_API_KEY,
    temperature=0.8,
    max_tokens=150,
)

# Initialize embedding model
embedder = SentenceTransformer(os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"))

# Initialize Qdrant client
qdrant_client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
    check_compatibility=False
)

# Typed dict for agent state
class AgentState(TypedDict):
    """
    State storage for the RAG orchestrator.

    Attributes:
        messages: Conversation history (BaseMessage sequence).
    """

    collection_name: str
    query: str
    messages: Annotated[Sequence[BaseMessage], add_messages]



@tool
def hybrid_search(query: str, collection_name: str, top_k: int = 3) -> List[str]:
    """
    Perform hybrid retrieval combining vector search over Qdrant.

    Args:
        query: The user's natural language query.
        collection_name: name of qdrant collection
        top_k: Number of top results to return.

    Returns:
        List of retrieved context strings.
    """ 
    print(f"\n🔍 Searching for: '{query}'")
    try:
        # 1. Convert the query text to a vector
        query_vector = embedder.encode(query).tolist()
        
        # 2. Use the 'query_points' method with the 'query' parameter
        search_result = qdrant_client.query_points(
            collection_name=collection_name,
            query=query_vector,
            limit=top_k
        )
        
        contexts = [point.payload['text'] for point in search_result.points]
        append_to_response([{"hybrid_search_success": contexts}], filename="graph_logs.json")
        
        return contexts or ["No relevant context found."]
    except Exception as e:
        error_message = f"An error occurred in hybrid_search: {str(e)}"

        append_to_response([{"hybrid_search_error": error_message}], filename="graph_logs.json")
        
        return [error_message]


# Bind hybrid_search tool to LLM
check_query_LLM = llm.bind_tools([hybrid_search])

# Intent classification node
def check_query_agent(state: AgentState) -> AgentState:
    """
    Classify user intent and select an appropriate tool or return an error.

    Calls:
      - hybrid_search for document retrieval.
    """
    system = SystemMessage(
        content=(
            "You are the RAG orchestrator which looks at query given by user after uploading the document of legal, insurance, contract, policy domains. Analyze the query and perform one of following task:"
            f"\n 1) Call hybrid_search Tool for document retrieval with collection_name: {state['collection_name']}"
            "\n 2) Return 'Ambiguous query' or 'Insufficient query' as response in format \n<response here> only and only when query is not creating any meaning."
            "\nDo not change query"
        )
    )
    query = state["query"]
    append_to_response([{"query_agent_in": query}], filename="graph_logs.json")

    response = check_query_LLM.invoke([system, query])

    append_to_response([{"query_agent_out": response}], filename="graph_logs.json")
    return {"messages": [response]}

# Content checking node
def check_docs_content(state: AgentState) -> AgentState:
    """
    Decide whether to expand the query or generate an answer based on retrieved contexts.
    """
    prompt = SystemMessage(
        content=(
            "You are a RAG assistant in legal, insurance, contract, policy domains which checks whether retrieved context answers user's query correctly and precisely."
            "\n• If context is insufficient or irrelevant, return 'expand_query' as response."
            "\n• If context fully answers, return 'answer_query' as response."
            "\n• If 'expand_query' was already returned as reponse once, then return 'answer_query' as response to avoid loops." 
            "\n make sure 'expand_query' is called only once in entire flow"
            f"\nRECENT CONVERSATION:\n{get_context(state,20)}"
            "\nResponse format must be like:\n<response here>"
        )
    )
    response = llm.invoke([prompt])
    append_to_response([{"check_content": response.content}], filename="graph_logs.json")
    return {"messages": [response]}

# Query expansion node
def expand_query(state: AgentState) -> AgentState:
    """
    Generate an optimized search query using conversation history.
    """
    system = SystemMessage(
        content=(
            "You are a query expansion assistant in legal, insurance, contract, policy domains. Produce exactly one optimized search query."
        )
    )
    human = HumanMessage(
        content=(f"RECENT CONVERSATION:\n{get_context(state,5)} "
                 "\nOutput query format must be like < Optimisied Query: <query here> >\n"
                 "\nDo not return anything else in the response")
    )
    response = llm.invoke([system, human])

    append_to_response([{"expand_query": response.content}], filename="graph_logs.json")
    return {"messages": [HumanMessage(content=f"New user query: {response.content}")]} 

# Answer generation node
def answer_query(state: AgentState) -> AgentState:
    """
    Integrate tool outputs and conversation history to generate the final answer.

    Returns:
        agent message with the answer.
    """
    prompt = SystemMessage(
        content=(
            "You are a RAG assistant in legal, insurance, contract, policy domains which buildis the final response to answer user query." 
            "\nGive to the point answer from retrieved context from hybrid_search tool only"
            "\nDo not drift away from main query "
            "\nDo not hallucinate and acknowledge any missing data."
            "\nIf not relevent context retrieved Tell the same to user"
            f"\nContext: {get_context(state,20)}"
            "\nResponse format must be like:\n<response here>"
        )
    )
    response = llm.invoke([prompt])
    append_to_response([{"answer_query": response.content}], filename="graph_logs.json")
    return {"messages": [response]}

# Instantiate graph
graph = StateGraph(AgentState)

# Register nodes
graph.add_node('check_query_agent', check_query_agent)
graph.add_node('hybrid_search_tool', ToolNode([hybrid_search]))
graph.add_node('check_docs_content', check_docs_content)
graph.add_node('expand_query', expand_query)
graph.add_node('answer_query', answer_query)

# Routing functions
def route_query(state: AgentState) -> str:
    last_msg = state["messages"][-1]
    calls = getattr(last_msg, "additional_kwargs", {}).get("tool_calls", [])
    
    if calls and calls[0].get("function", {}).get("name", "") == "hybrid_search":
        return "hybrid_search_tool"
    response = (last_msg.content or "").lower()
    
    if any(x in response for x in ("insufficient", "ambiguous")):
        return "answer_query"
    return "hybrid_search_tool"

def route_context(state: AgentState) -> str:
    response = (state["messages"][-1].content or "").lower()
    
    if "answer_query" in response:
        return "answer_query"
    
    if "expand_query" in response:
        return "expand_query"
    return "answer_query"

# Graph wiring
graph.add_edge(START, "check_query_agent")
graph.add_conditional_edges('check_query_agent', route_query, {'hybrid_search_tool': 'hybrid_search_tool', 'answer_query': 'answer_query'})
graph.add_edge("hybrid_search_tool","check_docs_content")
graph.add_conditional_edges(
    'check_docs_content',
    route_context,
    {
        'expand_query': 'expand_query',
        'answer_query': 'answer_query'
    }
)
graph.add_edge("expand_query","hybrid_search_tool")
graph.add_edge("answer_query",END)

# Compile graph into an app
app = graph.compile()

# Asynchronous orchestrator setup
async def process_query(query: str, collection_name: str) -> Tuple[str, str]:
    """
    Asynchronously process a single query through the graph with increased recursion limit.

    Returns:
        A tuple of (query, answer).
    """
    initial_state = AgentState({'messages': [HumanMessage(content=query)], "query":query, "collection_name":collection_name})
    # Use custom recursion limit for deep workflows
    result = await asyncio.to_thread(lambda: app.invoke(initial_state, config={"recursion_limit": 500}))
    return query, result['messages'][-1].content

async def parallel_orchestrator(queries: List[str], collection_name: str) -> Dict[str, str]:
    """
    Run multiple queries in parallel and return their answers.
    """
    tasks = [process_query(q,collection_name) for q in queries]
    completed = await asyncio.gather(*tasks)
    return dict(completed)

def graph_orchestrator_run(list_of_questions: List[str], collection_name: str)->Dict[str, str]:

    start = time.time()
    responses = asyncio.run(parallel_orchestrator(list_of_questions,collection_name))
    print("Graph Ended in: ",time.time()-start)

    return responses
