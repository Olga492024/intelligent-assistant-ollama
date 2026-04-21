import json
import os
from datetime import datetime
from .config import log_dir

class Logger:
    def __init__(self):
        os.makedirs(log_dir, exist_ok=True)

    def log(self, event_type: str, data: dict):
        timestamp = datetime.now().isoformat()
        entry = {
            "timestamp": timestamp,
            "event": event_type,
            "data": data
        }
        filename = f"{datetime.now().strftime('%Y-%m-%d')}.jsonl"
        filepath = os.path.join(log_dir, filename)
        with open(filepath, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")