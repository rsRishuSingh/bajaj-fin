# --- Imports and Setup ---
import sys
import os
import json
import requests
from typing import List, Annotated, Sequence, TypedDict, Dict

from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages
from langchain.docstore.document import Document

# utility.py contains these functions
from utility import append_to_response, get_context
import warnings

warnings.filterwarnings("ignore", category=FutureWarning, module="torch")
sys.setrecursionlimit(10**5)
load_dotenv()

# --- LLM and Tool Initialization ---
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


@tool
def get_from_endpoint(url: str) -> dict:
    """Makes a GET request to a URL and returns the JSON response."""
    print(f"--- Calling Tool: get_from_endpoint with URL: {url} ---")
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        return {"error": f"API request failed: {e}"}
    except json.JSONDecodeError:
        return {"error": "Failed to decode API response as JSON."}

tools = [get_from_endpoint]
llm_with_tools = llm.bind_tools(tools)


# --- Agent State Definition ---
class AgentState(TypedDict):
    """
    State storage for the dynamic agent.

    Attributes:
        query: The initial user query.
        chunks: Document chunks for context.
        goal: The high-level goal defined by the planner.
        steps: The list of steps to achieve the goal.
        current_step_index: Tracks the current step in the plan.
        routing_decision: The decision from the initial router node.
        messages: The sequence of messages forming the conversation history.
    """
    query: str
    chunks: List[Document]
    goal: str
    steps: List[dict]
    current_step_index: int
    routing_decision: str
    messages: Annotated[Sequence[BaseMessage], add_messages]


# --- Graph Nodes ---

def analyse_pdf_content(state: AgentState) -> Dict[str, str]:
    
    """Node 1: Initial router to decide between simple RAG or a complex agent."""
    
    query = state['query']
    chunks = state.get('chunks', [])

    if not chunks:
        return {"routing_decision": "rag_agent"}

    num_chunks = 20
    # Analyse first and last 20 chunks
    if len(chunks) <= num_chunks * 2:
        selected_chunks = chunks
    else:
        selected_chunks = chunks[:num_chunks] + chunks[-num_chunks:]

    full_content = "\n---\n".join([chunk.page_content for chunk in selected_chunks])

    prompt_content = f"""You are an expert routing agent. Your task is to analyze the user's query and the provided document content to decide the next step.
Your answer MUST be one of two choices and nothing else: "set_goal_agent" or "rag_agent".

RULES:
1. Return "set_goal_agent" if the document contains instructions for a multi-step process, API calls, or specific actions required to answer the user's query.
2. Return "rag_agent" if the query can likely be answered directly by reading the provided text without performing external actions.

---
DOCUMENT CONTENT (first 8000 characters):
{full_content[:8000]}

USER QUERY:
{query}
---
DECISION (must be "set_goal_agent" or "rag_agent"):"""

    response = llm.invoke([SystemMessage(content=prompt_content)])
    decision = response.content.strip().replace('"', '')
    if decision not in ["set_goal_agent", "rag_agent"]:
        decision = "rag_agent"  # Default to simple RAG on failure

    append_to_response([{"initial_routing_decision": decision}], filename="graph_logs.json")
    return {"routing_decision": decision}


def set_goal_agent(state: AgentState) -> Dict[str, any]:
    
    """Node 2: Planner that creates a goal and step-by-step plan."""
    
    query = state['query']
    chunks = state.get('chunks', [])
    full_content = "\n---\n".join([chunk.page_content for chunk in chunks])

    few_shot_example = """
    An example of the required JSON output format:
    {
      "goal": "Get the correct flight number",
      "steps": [
        {"step": 1, "description": "Call an API to get the initial city name."},
        {"step": 2, "description": "Use the PDF content to find the landmark for that city."},
        {"step": 3, "description": "Based on the landmark, determine the correct flight API to call."}
      ]
    }
    """

    prompt_content = f"""You are an expert planning agent. Analyze the document and query to generate a JSON object with a 'goal' and a list of 'steps' to achieve it. Your output MUST be only the JSON object.

{few_shot_example}
---
ANALYZE THE FOLLOWING:
DOCUMENT CONTENT (first 12000 characters):
{full_content[:12000]}

USER QUERY:
{query}
---
JSON OUTPUT:"""

    response = llm.invoke([SystemMessage(content=prompt_content)])
    cleaned_json = response.content.strip().removeprefix("```json").removesuffix("```").strip()

    try:
        plan = json.loads(cleaned_json)
        goal = plan.get("goal", "No goal defined.")
        steps = plan.get("steps", [])
    except (json.JSONDecodeError, AttributeError) as e:
        print(f"Error parsing plan: {e}")
        goal = "Failed to generate a valid plan."
        steps = []
        append_to_response([{"malformed_plan": response.content}], filename="graph_logs.json")

    return {"goal": goal, "steps": steps, "current_step_index": 0}


def perform_action(state: AgentState) -> Dict[str, List[BaseMessage]]:
    
    """Node 3: Executor that performs the current step by calling a tool."""
    
    current_step_index = state.get('current_step_index', 0)
    current_step = state['steps'][current_step_index]
    chunks = state.get('chunks', [])
    full_content = "\n---\n".join([chunk.page_content for chunk in chunks])
  
    prompt_content = f"""You are an executor agent. Your job is to carry out the current step of a plan.
Carefully analyze the overall goal and the specific instruction for the current step. Then, call the necessary tool with the correct parameters to complete ONLY the current step.

OVERALL GOAL: {state['goal']}
FULL PLAN: {json.dumps(state['steps'], indent=2)}
USER QUERY: {state['query']}
DOCUMENT CONTENT (first 12000 characters):
{full_content[:12000]}
---
CURRENT TASK (Step {current_step.get('step', current_step_index + 1)}): {current_step.get('description')}
---
Use your available tools. If you need information from a previous step, review the conversation history below."""
    
  
    messages_for_llm = [SystemMessage(content=prompt_content)] + state['messages']

    response = llm_with_tools.invoke(messages_for_llm)
    append_to_response([{"perform_action_invoked": {"step": current_step, "response": response.dict()}}], filename="graph_logs.json")
    
    return {"messages": [response]}


def check_completion(state: AgentState) -> str:
    
    """Node 4: Code-based router to check if all steps are complete."""
    
    current_step_index = state.get('current_step_index', 0)
    total_steps = len(state.get('steps', []))

    if current_step_index + 1 >= total_steps:
        return "answer_query"
    else:
        state['current_step_index'] = current_step_index + 1
        return "perform_action"


def answer_query(state: AgentState) -> Dict[str, List[BaseMessage]]:
    
    """Node 5: Final generator to synthesize the answer."""

    prompt_content = f"""You are an expert Q&A assistant. Your task is to provide a final, concise answer to the user's query based on the full conversation history. Synthesize all the information, including tool outputs, to formulate your response. Do not hallucinate. If the answer is present, provide it directly.

CONVERSATION HISTORY:
{get_context(state, 20)}

USER'S ORIGINAL QUERY:
{state['query']}
---
FINAL ANSWER:"""
    
    response = llm.invoke([SystemMessage(content=prompt_content)])
    append_to_response([{"final_answer": response.content}], filename="graph_logs.json")
    return {"messages": [response]}


# --- Graph Definition ---

# add nodes
workflow = StateGraph(AgentState)
workflow.add_node("analyse_pdf_content", analyse_pdf_content)
workflow.add_node("set_goal_agent", set_goal_agent)
workflow.add_node("perform_action", perform_action)
workflow.add_node("answer_query", answer_query)

# add tool nodes
tool_node = ToolNode(tools)
workflow.add_node("call_tool", tool_node)

# add the edges and conditional routing
workflow.set_entry_point("analyse_pdf_content")

# This conditional edge routes to the planner or a simple RAG agent (not defined)
workflow.add_conditional_edges(
    "analyse_pdf_content",
    lambda state: state['routing_decision'],
    {"set_goal_agent": "set_goal_agent", "rag_agent": "answer_query"} 
)

workflow.add_edge("set_goal_agent", "perform_action")
workflow.add_edge("perform_action", "call_tool")

# This conditional edge creates the main execution loop
workflow.add_conditional_edges(
    "call_tool",
    check_completion,
    {"perform_action": "perform_action", "answer_query": "answer_query"}
)

# The final node leads to the end of the graph
workflow.add_edge("answer_query", END)

# 3. Compile the graph
app = workflow.compile()

# To run this, you would invoke it with an initial state:
initial_state = {
    "query": "What is the flight number?",
    # "chunks": loaded_chunks, # Assuming chunks are loaded from a PDF
    "messages": []
}
result = app.invoke(initial_state)
print(result['messages'][-1].content)