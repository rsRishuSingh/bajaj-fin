import json
import requests
from typing import List, Dict, Annotated, Sequence, TypedDict

from langchain_core.messages import BaseMessage, SystemMessage, AIMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages

from .rag_system import MultiDocumentRAG
from .utils import get_context

# ORCHESTRATION LOGIC

class AgentState(TypedDict):
    query: str
    chunks: List[Dict]
    goal: str
    steps: List[Dict]
    current_step_index: int
    routing_decision: str
    messages: Annotated[Sequence[BaseMessage], add_messages]

@tool
def get_from_endpoint(url: str) -> dict:
    """Makes a GET request to a URL and returns the JSON response. Use this to call APIs mentioned in the document."""
    print(f"--- AGENT TOOL CALL: GET {url} ---")
    try:
        response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        return {"error": f"API request failed: {e}"}
    except json.JSONDecodeError:
        return {"error": f"Failed to decode API response as JSON. Content: {response.text}"}

class Orchestrator:
    def __init__(self, rag_system: MultiDocumentRAG):
        self.rag_system = rag_system
        self.llm = rag_system.llm
        self.tools = [get_from_endpoint]
        self.llm_with_tools = self.llm.bind_tools(self.tools)
        self.app = self._build_graph()

    def _build_graph(self):
        workflow = StateGraph(AgentState)
        workflow.add_node("analyse_document_content", self.analyse_document_content)
        workflow.add_node("rag_agent", self.rag_agent)
        workflow.add_node("set_goal_agent", self.set_goal_agent)
        workflow.add_node("perform_action", self.perform_action)
        workflow.add_node("call_tool", ToolNode(self.tools))
        workflow.add_node("increment_step", self.increment_step)
        workflow.add_node("answer_query", self.answer_query)
        
        workflow.set_entry_point("analyse_document_content")
        workflow.add_conditional_edges("analyse_document_content", lambda s: s['routing_decision'])
        workflow.add_edge("rag_agent", "answer_query")
        workflow.add_conditional_edges("set_goal_agent", lambda s: "end_with_error" if not s.get("steps") else "continue_to_action", {
            "continue_to_action": "perform_action",
            "end_with_error": "answer_query"
        })
        workflow.add_edge("perform_action", "call_tool")
        workflow.add_conditional_edges("call_tool", lambda s: "end_loop" if s['current_step_index'] + 1 >= len(s.get('steps', [])) else "continue_loop", {
            "continue_loop": "increment_step",
            "end_loop": "answer_query"
        })
        workflow.add_edge("increment_step", "perform_action")
        workflow.add_edge("answer_query", END)
        return workflow.compile()

    async def analyse_document_content(self, state: AgentState) -> Dict:
        """
        Intelligently routes based on both the document's content and the user's query.
        It safely uses an LLM to determine the user's intent only when necessary.
        """
        print("--- Node: analyse_document_content ---")
        
        print("   - Analyzing query intent with LLM...")
        query = state['query']
        
        prompt = f"""You are an intelligent router. A document or a web page has been provided that may contain instructions or API endpoints. Your task is to analyze the user's query and decide the best way to answer it.

You have two choices:
1. 'rag_agent': Use this for general questions about the document's content, purpose, or for summarizing its text. For example: "What is this document about?", "Summarize the introduction".
2. 'set_goal_agent': Use this if the user's query requires you to FOLLOW instructions, EXECUTE a plan, or INTERACT with a URL or API. For example: "What is the flight number?", "Go to the link and get the token".

User's Query: "{query}"

Based on the user's query, which agent is the most appropriate? Your response must be ONLY 'rag_agent' or 'set_goal_agent'.
Decision:"""

        try:
            response = await self.llm.ainvoke(prompt)
            decision = response.content.strip()
            if decision not in ["rag_agent", "set_goal_agent"]:
                print(f"   - LLM returned an invalid decision: '{decision}'. Defaulting to rag_agent.")
                decision = "rag_agent"
        except Exception as e:
            print(f"   - LLM routing failed: {e}. Defaulting to rag_agent.")
            decision = "rag_agent"

        print(f"Routing Decision: {decision}")
        return {"routing_decision": decision}
    
    async def rag_agent(self, state: AgentState) -> Dict:
        print("--- Node: rag_agent ---")
        answer = await self.rag_system.answer_question(state['query'])
        return {"messages": [AIMessage(content=answer)]}
    
    async def set_goal_agent(self, state: AgentState) -> Dict:
        print("--- Node: set_goal_agent ---")
        content = "\n".join([c['text'] for c in state['chunks']])
        example = """{"goal": "Find the flight number.","steps": [{"step": 1,"description": "Call the API at https://.../myFavouriteCity to find the city name."}, {"step": 2,"description": "Find the landmark associated with that city using the document text."}, {"step": 3,"description": "Based on the landmark, determine the correct flight API from the document and call it."}]}"""
        prompt = f"""Create a JSON plan to solve the user's query using the document. Each step must be a dictionary with 'step' and 'description'. Output ONLY the JSON.

Document:
{content}

Query: "{state['query']}"

Example:
{example}

JSON Plan:"""
        response = await self.llm.ainvoke(prompt)
        try:
            plan = json.loads(response.content.strip().removeprefix("```json").removesuffix("```"))
            return {"goal": plan.get("goal"), "steps": plan.get("steps"), "current_step_index": 0}
        except json.JSONDecodeError:
            return {"goal": "Failed to create a plan.", "steps": [], "messages": [AIMessage(content="I could not create a valid plan to solve this.")]}
    
    async def perform_action(self, state: AgentState) -> Dict:
        idx = state['current_step_index']
        step = state['steps'][idx]
        description = step.get('description', str(step))
        print(f"--- Node: perform_action (Step {idx+1}: {description}) ---")
        content = "\n".join([c['text'] for c in state['chunks']])
        history = get_context(state)
        prompt = f"""You are an executor agent. Your task is to perform a single step of a plan.
Full Document Content:
{content}
Conversation History:
{history}
---
Current Task: {description}
---
Execute this task. If the task is to find information in the document, state the information clearly. If the task requires calling an API, use the available tool."""
        response = await self.llm_with_tools.ainvoke([SystemMessage(prompt)])
        return {"messages": [response]}

    async def increment_step(self, state: AgentState) -> Dict:
        return {"current_step_index": state['current_step_index'] + 1}

    async def answer_query(self, state: AgentState) -> Dict:
        print("--- Node: answer_query ---")
        if not state.get('messages'):
            return {"messages": [AIMessage(content="I apologize, but I was unable to complete the request.")]}
        if len(state.get('steps', [])) == 0:
             return {"messages": state['messages']}

        history = get_context(state)
        prompt = f"""Synthesize the conversation history into a final, direct answer to the user's original query.

History:
{history}

Original Query: "{state['query']}"

Final Answer:"""
        response = await self.llm.ainvoke(prompt)
        return {"messages": [response]}