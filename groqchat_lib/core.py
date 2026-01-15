from typing import List, Dict, Optional, Any
from groq import Groq
from .session_manager import SessionManager
from .history_manager import HistoryHandler
from .rag_engine import RAGEngine
from .audit_logger import AuditLogger
from .utils import get_logger
from .exceptions import SessionError

class GroqChat:
    def __init__(
        self,
        api_key: str,
        model: str = "llama-3.1-8b-instant",
        max_tokens: int = 8192,
        sessions_dir: str = "sessions",
        summarizer_model: Optional[str] = None,
        log_level: int = 20,
        log_file: Optional[str] = None,
        rag_chunk_size: int = 500,
        rag_chunk_overlap: int = 50,
        rag_top_k: int = 3,
    ):
        self.client = Groq(api_key=api_key)
        self.model = model
        self.max_tokens = max_tokens
        self.logger = get_logger("GroqChat", log_level, log_file)
        effective_summarizer = summarizer_model or model
        self.session_manager = SessionManager(sessions_dir)
        self.history_handler = HistoryHandler(self.client, max_tokens, effective_summarizer)
        self.rag_engine = RAGEngine(self.client, model, max_tokens, rag_chunk_size, rag_chunk_overlap, rag_top_k)
        self.audit_logger = AuditLogger(sessions_dir)
        self.logger.info(f"GroqChat инициализирован. Сессии: {self.session_manager.sessions_dir}")

    # === Сессии ===
    def new_session(self, name: str):
        self.session_manager.new_session(name)
        self.audit_logger.set_session(name)

    def load_session(self, name: str):
        self.session_manager.load_session(name)
        self.audit_logger.set_session(name)

    def save_session(self, name: Optional[str] = None):
        self.session_manager.save_session(name)

    def delete_session(self, name: str) -> bool:
        return self.session_manager.delete_session(name)

    def list_sessions(self) -> List[str]:
        return self.session_manager.list_sessions()

    @property
    def current_session_name(self) -> Optional[str]:
        return self.session_manager.current_session_name

    @property
    def messages(self) -> List[Dict[str, str]]:
        return self.session_manager.messages

    # === Диалог ===
    def get_answer(self, prompt: str) -> str:
        if not self.session_manager.is_active:
            raise SessionError("Сессия не выбрана")
        self.audit_logger.log_chat_message("user", prompt)
        self.session_manager.messages.append({"role": "user", "content": prompt})
        tokens = self.history_handler.count_context_tokens(self.session_manager.messages)
        if tokens >= self.max_tokens:
            self.session_manager.messages = self.history_handler.summarize_old_messages(self.session_manager.messages)

        # Внутри get_answer, перед вызовом API:
        print(">>> Отправляемая история:")
        for i, msg in enumerate(self.session_manager.messages):
            print(f"{i}: {msg['role']}: {msg['content'][:60]}...")

        response = self.client.chat.completions.create(
            model=self.model,
            messages=self.session_manager.messages,
            temperature=0.7,
            max_tokens=1024
        )
        answer = response.choices[0].message.content
        self.session_manager.messages.append({"role": "assistant", "content": answer})
        self.session_manager.save_session()
        self.audit_logger.log_chat_message("assistant", answer)
        return answer

    # === RAG ===
    def rag_query_from_text(self, prompt: str, source_text: str, strict_context: bool = True) -> str:
        context = self.rag_engine._prepare_context(source_text, prompt)
        answer = self.rag_engine.query(prompt, context, strict_context)
        self.audit_logger.log_rag_query(
            prompt=prompt,
            context=context,
            answer=answer,
            strict_context=strict_context,
            source_text_length=len(source_text)
        )
        return answer

    def rag_query_from_file(self, prompt: str, file_path: str, strict_context: bool = True) -> str:
        source_text = self.rag_engine._load_text_from_file(file_path)
        context = self.rag_engine._prepare_context(source_text, prompt)
        answer = self.rag_engine.query(prompt, context, strict_context)
        self.audit_logger.log_rag_query(
            prompt=prompt,
            context=context,
            answer=answer,
            strict_context=strict_context,
            source_file=str(file_path),
            source_text_length=len(source_text)
        )
        return answer

    # === Модели ===
    def fetch_available_models(self, save_to: Optional[str] = None) -> List[Dict[str, Any]]:
        try:
            import httpx
            headers = {"Authorization": f"Bearer {self.client.api_key}"}
            response = httpx.get("https://api.groq.com/openai/v1/models", headers=headers)
            response.raise_for_status()
            raw_data = response.json()
            models = raw_data.get("data", [])
            enriched = []
            for m in models:
                meta = m.get("metadata", {})
                enriched.append({
                    "id": m.get("id"),
                    "object": m.get("object"),
                    "created": m.get("created"),
                    "owned_by": m.get("owned_by"),
                    "active": m.get("active", True),
                    "context_window": meta.get("context_window") if isinstance(meta, dict) else None,
                    "max_output_tokens": meta.get("max_output_tokens") if isinstance(meta, dict) else None,
                })
            if save_to:
                import json as json_mod
                with open(save_to, "w", encoding="utf-8") as f:
                    json_mod.dump(enriched, f, ensure_ascii=False, indent=2)
                self.logger.info(f"Модели сохранены в: {save_to}")
            return enriched
        except Exception as e:
            self.logger.error(f"Ошибка получения моделей: {e}")
            raise