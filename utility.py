import re
import os
import json
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from typing import  Any, List, Union
from chunking import recursive_split
from datetime import datetime, timezone, timedelta
from langchain_core.messages import HumanMessage, BaseMessage

load_dotenv()
MODEL_NAME       = os.getenv("MODEL_NAME", "qwen/qwen3-32b")

def _unwrap(item: Any) -> Any:
    """
    Recursively convert BaseMessage objects to dicts via model_dump(),
    and leave other types (primitives, lists, dicts) intact.
    """
    if isinstance(item, BaseMessage):
        return item.model_dump()
    elif isinstance(item, dict):
        return {k: _unwrap(v) for k, v in item.items()}
    elif isinstance(item, list):
        return [_unwrap(v) for v in item]
    else:
        return item

def append_to_response(
    new_items: List[Union[dict, BaseMessage, Any]],
    filename: str = "response.json"
) -> None:
    """
    Append a list of items to a JSON array in `filename`, tagging each with a 'timestamp'.
    Supports dicts, lists, primitives, and LangChain Message objects (BaseMessage).
    """
    # Indian timezone
    IST = timezone(timedelta(hours=5, minutes=30))
    now = datetime.now(IST).isoformat()

    # Load existing data (or start fresh list)
    if os.path.exists(filename):
        with open(filename, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
                if not isinstance(data, list):
                    raise ValueError(f"{filename} does not contain a JSON list.")
            except (json.JSONDecodeError, ValueError):
                data = []
    else:
        data = []

    # Process and append each new item
    for raw in new_items:
        # First unwrap any nested BaseMessage / lists / dicts
        item_dict = _unwrap(raw)

        # Must end up as a dict or primitive
        if not isinstance(item_dict, dict):
            # wrap primitives under a generic key
            item_dict = {"value": item_dict}

        # add timestamp if missing
        item_dict.setdefault("timestamp", now)
        data.append(item_dict)

    # Write back
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def remove_think(text: str) -> str:
    """
    Removes the <think> tag and all content inside it from the input text.
    
    Parameters:
    text (str): The input text that may contain <think>...</think>

    Returns:
    str: Text with the <think> block removed.
    """
    return re.sub(r'<think>.*?</think>\n?', '', text, flags=re.DOTALL)


def get_context(state, num_messages: int = 10) -> str:
    """
    Builds a well-structured context string from the last `num_messages`
    in state["messages"], including content, hidden reasoning, and tool calls.

    Args:
        state: AgentState with "messages" list of BaseMessage objects.
        num_messages: Number of recent messages to include (default 10).

    Returns:
        A formatted multi-line string representing the conversation history.
    """
    ctx_entries: List[str] = []

    # Take the last num_messages items
    count = len(state["messages"])
    num_messages = min(count,num_messages)
    recent = state["messages"][-num_messages:]

    for msg in recent:
        # 1) Determine speaker label
        msg_type = getattr(msg, "type",
                           msg.__class__.__name__.replace("Message", "").lower())
        speaker = msg_type.title()  # e.g. "Human", "Ai", "Tool"

        # 2) Main content
        content = getattr(msg, "content", "<no content>") or "<no content>"
        entry_lines = [f"{speaker} Content: {content}"]

        ak = getattr(msg, "additional_kwargs", {}) or {}
        # 3) Tool calls
        tool_calls = ak.get("tool_calls") or []
        for call in tool_calls:
            fn = call["function"]["name"]
            # Merge positional args and keyword args
            args = call.get("args", []) or []
            kwargs = call.get("kwargs", {}) or {}
            args_repr = ", ".join(
                [repr(a) for a in args] +
                [f"{k}={v!r}" for k, v in kwargs.items()]
            )
            entry_lines.append(f"Tool Call: {fn}({args_repr})")

        # Combine this message block
        ctx_entries.append("\n".join(entry_lines))

    context = remove_think("\n\n---\n\n".join(ctx_entries))

    return context


def compress_context(state:Any) -> str:
    """
    Compresses the conversation context by retrieving the latest messages,
    splitting them into manageable chunks, and aggregating LLM-generated summaries.

    Parameters:
        state: AgentState or similar object containing conversation history.

    Returns:
        A single string representing the compressed context,
        built by concatenating LLM responses for each chunk.
    """
    # Retrieve the most recent 10 messages from state
    if len(state) == 0:
        return ""
    context = get_context(state, 10)

    # Split the context into overlapping chunks of ~500 tokens with 100 token overlap
    chunks = recursive_split(context, 2000, 200)

    llm = ChatGroq(model=MODEL_NAME)
    compressed = ""

    # Prompt to guide the LLM on how to compress each chunk
    final_prompt = (
        "Please produce a concise and clear summary of the following conversation excerpt without any redudancy."
        "focusing on key insights, company details, context and tools. "
        "Maintain coherence across chunks when concatenated."
        "Compress all context to maximum of 200 chunks size"
    )

    for chunk in chunks:
        msg = HumanMessage(content=chunk)
        # Invoke LLM with prompt + chunk
        response = llm.invoke([HumanMessage(content=final_prompt), msg])
        final_response = remove_think(response.content)
        compressed += final_response + "\n"

    return compressed.strip()
