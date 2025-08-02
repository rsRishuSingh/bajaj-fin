import re
import os
import json
from dotenv import load_dotenv
from typing import  Any, List, Union,Dict
from datetime import datetime, timezone, timedelta
from langchain_core.messages import BaseMessage

load_dotenv()


def save_responses_append(
    responses: List[Dict[str, Union[int, str]]],
    filename: str = "responses.json"
) -> None:
    """
    Appends a new set of responses to a JSON file.

    This function reads the existing JSON file, adds the new list of responses
    as a new entry in the main dictionary, and saves it back.

    Args:
        responses: A list of dictionaries representing the new response set to add.
        filename:  Path to the JSON file.
    """
    all_responses = {}
    # 1. Load existing data if the file exists and is valid
    if os.path.exists(filename):
        try:
            with open(filename, "r", encoding="utf-8") as f:
                existing_data = json.load(f)
                # Ensure we are working with a dictionary
                if isinstance(existing_data, dict):
                    all_responses = existing_data
        except (json.JSONDecodeError, FileNotFoundError):
            # If file is corrupt or unreadable, we'll start fresh
            pass

    # 2. Determine the key for the new response set
    new_key = f"response_{len(all_responses) + 1}"

    # 3. Add the new response list under the new key
    all_responses[new_key] = responses

    # 4. Write the entire updated dictionary back to the file
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(all_responses, f, ensure_ascii=False, indent=4)

    print(f"✅ Appended new entry '{new_key}' to {filename}")



def load_responses(filename: str = "responses.json") -> Dict[str, List[Dict[str, Union[int, str]]]]:
    """
    Loads the dictionary of all response sets from the JSON file.

    This function returns the data in the exact format it is saved, which is a
    dictionary where each key maps to a specific response set.

    Args:
        filename: Path (and name) of the file to read.

    Returns:
        A dictionary containing all saved response sets.
        Returns an empty dictionary if the file is not found or is invalid.
    """
    try:
        with open(filename, "r", encoding="utf-8") as f:
            # Load the entire dictionary from the file
            all_responses = json.load(f)

            # Validate that the loaded data is a dictionary
            if not isinstance(all_responses, dict):
                print(f"❌ Expected a dictionary in {filename}, but found {type(all_responses).__name__}. Returning empty.")
                return {}

        print(f"✅ Loaded {len(all_responses)} response sets from {filename}")
        return all_responses

    except FileNotFoundError:
        print(f"⚠️ File not found: {filename}. Returning an empty dictionary.")
        return {}
    except json.JSONDecodeError:
        print(f"❌ Error decoding JSON from {filename}. File might be corrupt. Returning an empty dictionary.")
        return {}
    

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
    filename: str = "graph_logs.json"
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

