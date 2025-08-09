# flow_director.py

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
from chunking import process_and_chunk_pdf
from utility import append_to_response, get_context
import warnings

warnings.filterwarnings("ignore", category=FutureWarning, module="torch")
sys.setrecursionlimit(10**5)
load_dotenv()


AZURE_ENDPOINT        = "https://rishu-mdodjz43-eastus2.cognitiveservices.azure.com/"
AZURE_DEPLOYMENT_NAME = "gpt-4o"
AZURE_API_VERSION     = "2024-05-01-preview"
AZURE_API_KEY         = os.getenv("AZURE_OPENAI_GPT_API")

llm = AzureChatOpenAI(
    deployment_name=AZURE_DEPLOYMENT_NAME,
    azure_endpoint=AZURE_ENDPOINT,
    api_version=AZURE_API_VERSION,
    api_key=AZURE_API_KEY,
    temperature=0.2,
    max_tokens=2048,
)
@tool
def get_from_endpoint(url: str) -> dict:
    """Makes a GET request to a URL and returns the JSON response."""
    print(f"--- Calling Tool: get_from_endpoint with URL: {url} ---")
    try:
        response = requests.get(url)
        response.raise_for_status()
        # The log for the tool call itself is better placed here.
        log_data = {"url": url, "response": response.json()}
        append_to_response([{"tool_call_get_from_endpoint": log_data}], filename="graph_logs.json")
        return response.json()
    except requests.RequestException as e:
        return {"error": f"API request failed: {e}"}
    except json.JSONDecodeError:
        return {"error": "Failed to decode API response as JSON."}

tools = [get_from_endpoint]
llm_with_tools = llm.bind_tools(tools)

# Agent State
class AgentState(TypedDict):
    query: str
    chunks: List[Document]
    goal: str
    steps: List[dict]
    current_step_index: int
    routing_decision: str
    messages: Annotated[Sequence[BaseMessage], add_messages]

# Graph Nodes

def analyse_pdf_content(state: AgentState) -> Dict[str, str]:
    """Node 1: Initial router to decide between simple RAG or a complex agent."""
    query = state['query']
    chunks = state.get('chunks', [])
    if not chunks: return {"routing_decision": "rag_agent"}

    selected_chunks = chunks[:20] + chunks[-20:] if len(chunks) > 40 else chunks
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
    append_to_response([{"analyse_pdf_content": response}], filename="graph_logs.json")

    decision = response.content.strip().replace('"', '')
    if decision not in ["set_goal_agent", "rag_agent"]:
        decision = "rag_agent"
    return {"routing_decision": decision}

def set_goal_agent(state: AgentState) -> Dict[str, any]:
    """Node 2: Planner that creates a goal and step-by-step plan."""
    query = state['query']
    chunks = state.get('chunks', [])
    full_content = "\n---\n".join([chunk.page_content for chunk in chunks])
    few_shot_example = """
    {
      "goal": "Get the correct flight number",
      "steps": [
        {"step": 1, "description": "Call an API to get the initial city name."},
        {"step": 2, "description": "Use the PDF content to find the landmark for that city."},
        {"step": 3, "description": "Based on the landmark, determine the correct flight API to call."}
      ]
    }"""
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
    append_to_response([{"set_goal_agent": response}], filename="graph_logs.json")

    cleaned_json = response.content.strip().removeprefix("```json").removesuffix("```").strip()
    try:
        plan = json.loads(cleaned_json)
        goal, steps = plan.get("goal", "No goal defined."), plan.get("steps", [])
    except (json.JSONDecodeError, AttributeError) as e:
        print(f"Error parsing plan: {e}")
        goal, steps = "Failed to generate a valid plan.", []
    return {"goal": goal, "steps": steps, "current_step_index": 0}

def check_plan_validity(state: AgentState) -> str:
    """A function for conditional routing. Checks if the plan is valid."""
    return "continue_to_action" if state.get("steps") else "end_with_error"

def perform_action(state: AgentState) -> Dict[str, List[BaseMessage]]:
    """Node 3: Executor that performs the current step by calling a tool."""
    idx = state.get('current_step_index', 0)
    step = state['steps'][idx]
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
CURRENT TASK (Step {step.get('step', idx + 1)}): {step.get('description')}
---
Use your available tools. If you need information from a previous step, review the conversation history below."""
    messages = [SystemMessage(content=prompt_content)] + state['messages']
    response = llm_with_tools.invoke(messages)
    # The log for the LLM's decision to call a tool
    append_to_response([{"perform_action_decision": response.model_dump()}], filename="graph_logs.json")
    return {"messages": [response]}


def should_continue(state: AgentState) -> str:
    """This is a read-only function for conditional routing that decides if the loop should continue."""
    if state['current_step_index'] + 1 >= len(state.get('steps', [])):
        return "end_loop"
    else:
        return "continue_loop"

def increment_step_index(state: AgentState) -> Dict[str, int]:
    """This is a dedicated node to reliably update the step counter."""
    return {"current_step_index": state.get('current_step_index', 0) + 1}

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
workflow = StateGraph(AgentState)
workflow.add_node("analyse_pdf_content", analyse_pdf_content)
workflow.add_node("set_goal_agent", set_goal_agent)
workflow.add_node("perform_action", perform_action)
workflow.add_node("call_tool", ToolNode(tools))
workflow.add_node("increment_step", increment_step_index) # Added new state updater node
workflow.add_node("answer_query", answer_query)

workflow.set_entry_point("analyse_pdf_content")
workflow.add_conditional_edges("analyse_pdf_content", lambda s: s['routing_decision'], {"set_goal_agent": "set_goal_agent", "rag_agent": "answer_query"})
workflow.add_conditional_edges("set_goal_agent", check_plan_validity, {"continue_to_action": "perform_action", "end_with_error": "answer_query"})

# --- FIX: Corrected the infinite loop with an explicit, robust state update step ---
workflow.add_edge("perform_action", "call_tool")
workflow.add_conditional_edges(
    "call_tool",
    should_continue, # The read-only decision function
    {
        "continue_loop": "increment_step", # If continuing, first increment the counter
        "end_loop": "answer_query"         # If finished, go to the final answer
    }
)
# After the index is incremented, loop back to perform the next action
workflow.add_edge("increment_step", "perform_action")

workflow.add_edge("answer_query", END)
app = workflow.compile()

# --- Main Execution Logic ---
def load_docs(filepath: str = "all_docs.json") -> List[Document]:
    if not os.path.exists(filepath): return []
    print(f"📤 Loading chunks from {filepath}")
    with open(filepath, "r", encoding="utf-8") as f:
        return [Document(page_content=d["page_content"], metadata=d["metadata"]) for d in json.load(f)]

def save_docs(docs: List[Document], filepath: str = "all_docs.json") -> None:
    print(f"📥 Saving {len(docs)} chunks → {filepath}")
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(
            [{"page_content": d.page_content, "metadata": d.metadata} for d in docs],
            f, indent=2, ensure_ascii=False
        )

docs = load_docs()
if not docs:
    docs = process_and_chunk_pdf("level4.pdf", chunk_size=1000, chunk_overlap=150)
    if docs: save_docs(docs)

if docs:
    initial_state = {"query": "What is the flight number?", "chunks": docs, "messages": []}
    result = app.invoke(initial_state, {"recursion_limit": 50})
    print("--- FINAL ANSWER ---")
    print(result['messages'][-1].content)
else:
    print("Could not load or process documents. Halting execution.")