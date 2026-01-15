import json
from pathlib import Path
from datetime import datetime
from typing import Optional

class AuditLogger:
    def __init__(self, sessions_dir="sessions"):
        self.sessions_dir = Path(sessions_dir)
        self.sessions_dir.mkdir(exist_ok=True)
        self.current_session: Optional[str] = None
        self.audit_path: Optional[Path] = None

    def set_session(self, name: str):
        self.current_session = name
        self.audit_path = self.sessions_dir / f"{name}_audit.json"

    def _log_event(self, event_type: str,  dict):
        if not self.audit_path:
            return
        entry = {"timestamp": datetime.utcnow().isoformat() + "Z", "event_type": event_type, **dict}
        logs = json.loads(self.audit_path.read_text(encoding="utf-8")) if self.audit_path.exists() else []
        logs.append(entry)
        self.audit_path.write_text(json.dumps(logs, ensure_ascii=False, indent=2), encoding="utf-8")

    def log_chat_message(self, role: str, content: str):
        self._log_event("chat_message", {"role": role, "content": content})

    def log_rag_query(self, prompt: str, context: str, answer: str, strict_context: bool,
                      source_file: Optional[str] = None, source_text_length: Optional[int] = None):
        self._log_event("rag_query", {
            "prompt": prompt,
            "context_used": context,
            "answer": answer,
            "strict_context": strict_context,
            "source_file": source_file,
            "source_text_length": source_text_length
        })