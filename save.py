import json
from typing import Dict

def save_responses(responses: Dict[str, str], filename: str = "responses.json") -> None:
    """
    Save the given query→answer mapping as a pretty-printed JSON file.

    Args:
        responses: Dict where keys are the original queries and values are the answers.
        filename:  Path (and name) of the file to write.
    """
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(responses, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(responses)} responses to {filename}")