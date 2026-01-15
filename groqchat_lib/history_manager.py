from typing import List, Dict, Optional
from groq import Groq
from .utils import count_messages_tokens, get_tokenizer

class HistoryHandler:
    def __init__(self, client: Groq, max_tokens=8192, summarizer_model="llama-3.1-8b-instant"):
        self.client = client
        self.max_tokens = max_tokens
        self.summarizer_model = summarizer_model
        self.tokenizer = get_tokenizer()

    def count_tokens(self, text: str) -> int:
        return count_tokens(self.tokenizer, text)

    def count_context_tokens(self, messages: List[Dict[str, str]]) -> int:
        return count_messages_tokens(self.tokenizer, messages)

    def summarize_old_messages(self, messages: List[Dict[str, str]], buffer_ratio=0.75):
        target_max = int(self.max_tokens * buffer_ratio)
        msgs = messages.copy()
        while self.count_context_tokens(msgs) > target_max and len(msgs) > 2:
            to_sum = [msgs[0], msgs[1]]
            summary_text = "\n".join(f"{m['role']}: {m['content']}" for m in to_sum)
            try:
                resp = self.client.chat.completions.create(
                    model=self.summarizer_model,
                    messages=[
                        {"role": "system", "content": "Кратко обобщи диалог в одно предложение."},
                        {"role": "user", "content": summary_text}
                    ],
                    temperature=0.3,
                    max_tokens=64
                )
                summary = resp.choices[0].message.content.strip()
                new_msg = {"role": "assistant", "content": f"[Ранее: {summary}]"}
                msgs = [new_msg] + msgs[2:]
            except Exception:
                break
        return msgs