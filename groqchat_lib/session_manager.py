import json
from pathlib import Path
from typing import List, Optional, Dict
from .utils import sanitize_filename
from .exceptions import SessionError

class SessionManager:
    def __init__(self, sessions_dir="sessions"):
        self.sessions_dir = Path(sessions_dir)
        self.sessions_dir.mkdir(exist_ok=True)
        self.current_session_name: Optional[str] = None
        self.messages: List[Dict[str, str]] = []

    def _get_path(self, name):
        return self.sessions_dir / f"{sanitize_filename(name)}.json"

    def list_sessions(self):
        return sorted(f.stem for f in self.sessions_dir.glob("*.json"))

    def load_session(self, name):
        path = self._get_path(name)
        if not path.exists():
            self.messages = []
            self.current_session_name = name
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                for msg in data:
                    assert isinstance(msg, dict) and "role" in msg and "content" in msg
                self.messages = data
            self.current_session_name = name
        except Exception as e:
            raise SessionError(f"Ошибка загрузки сессии '{name}': {e}")

    def save_session(self, name=None):
        name = name or self.current_session_name
        if not name:
            raise SessionError("Имя сессии не задано")
        with open(self._get_path(name), "w", encoding="utf-8") as f:
            json.dump(self.messages, f, ensure_ascii=False, indent=2)

    def new_session(self, name):
        self.messages = []
        self.current_session_name = name

    def new_session(self, name: str):
        self.messages = [
            {"role": "system",
             "content": "Ты — помощник, который помнит весь текущий диалог. Отвечай с учётом всей истории. Если в дальнейших запросах будут обращения к прошлым диалогам, обязательно используй информацию из истории."}
        ]
        self.current_session_name = name

    def delete_session(self, name):
        path = self._get_path(name)
        existed = path.exists()
        if existed:
            path.unlink()
            if self.current_session_name == name:
                self.current_session_name = None
                self.messages = []
        return existed

    @property
    def is_active(self):
        return self.current_session_name is not None