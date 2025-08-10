import os
import json
from datetime import datetime
import pytz

def get_ist_timestamp():
    ist = pytz.timezone("Asia/Kolkata")
    now = datetime.now(ist)
    return now.strftime("%d-%m-%Y_%H-%M-%S")

def add_timestamp_to_data(data):
    """Insert timestamp at the start of the dictionary."""
    if isinstance(data, dict):
        return {"timestamp": get_ist_timestamp(), **data}
    else:
        return {"timestamp": get_ist_timestamp(), "data": data}

def append_log(dir_path, filename, data):
    os.makedirs(dir_path, exist_ok=True)
    filepath = os.path.join(dir_path, filename)
    # Read existing logs if file exists, else start with empty list
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            try:
                file_content = f.read().replace('\u00A0', ' ')
                logs = json.loads(file_content)
            except json.JSONDecodeError:
                logs = []
    else:
        logs = []
    # Append new log (data already has timestamp inside)
    logs.append(data)
    # Write back to file
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(logs, f, indent=4, ensure_ascii=False)

def log_incoming_request(input_data):
    dir_path = os.path.join(os.getcwd(), "status")
    data_with_timestamp = add_timestamp_to_data(input_data)
    append_log(dir_path, "incoming.json", data_with_timestamp)
    print(f"📥 Incoming request logged to: {os.path.join(dir_path, 'incoming.json')}")

def log_and_save_response(output_data, is_success):
    status = "success" if is_success else "fail"
    dir_path = os.path.join(os.getcwd(), "status")
    filename = f"{status}.json"
    try:
        data_with_timestamp = add_timestamp_to_data(output_data)
        append_log(dir_path, filename, data_with_timestamp)
        print(f"✅ Results successfully saved to: {os.path.join(dir_path, filename)}")
    except Exception as e:
        # Log the error in the status directory
        error_data = {
            "error": str(e),
            "output_data": output_data
        }
        error_dir = os.path.join(os.getcwd(), "status")
        error_with_timestamp = add_timestamp_to_data(error_data)
        append_log(error_dir, "fail.json", error_with_timestamp)
        print(f"❌ Failed to save results. Error logged to: {os.path.join(error_dir, 'fail.json')}")