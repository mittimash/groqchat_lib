from pathlib import Path
from typing import List, Dict, Union
from groq import Groq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer, util
import torch, fitz, docx, json as json_mod
from .utils import get_tokenizer, count_tokens
from .exceptions import RAGError

class RAGEngine:
    def __init__(self, client: Groq, model="llama-3.1-8b-instant", max_tokens=8192,
                 chunk_size=500, chunk_overlap=50, top_k=3):
        self.client = client
        self.model = model
        self.max_tokens = max_tokens
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k
        self.tokenizer = get_tokenizer()
        self._embedder = None

    @property
    def embedder(self):
        if self._embedder is None:
            self._embedder = SentenceTransformer('all-MiniLM-L6-v2')
        return self._embedder

    def _load_text_from_file(self, path: Union[str, Path]) -> str:
        p = Path(path)
        suffix = p.suffix.lower()
        try:
            if suffix == ".txt":
                return p.read_text(encoding="utf-8")
            elif suffix == ".json":
                with open(p, "r", encoding="utf-8") as f:
                    data = json_mod.load(f)
                return json_mod.dumps(data, ensure_ascii=False, indent=2)
            elif suffix == ".pdf":
                doc = fitz.open(p)
                text = "".join(page.get_text() for page in doc)
                doc.close()
                return text
            elif suffix in (".docx", ".doc"):
                doc = docx.Document(p)
                return "\n".join(paragraph.text for paragraph in doc.paragraphs)
            else:
                raise RAGError(f"Неподдерживаемый формат: {suffix}")
        except Exception as e:
            raise RAGError(f"Ошибка загрузки {path}: {e}")

    def _prepare_context(self, source_text: str, query: str) -> str:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=lambda x: count_tokens(self.tokenizer, x),
        )
        chunks = splitter.split_text(source_text)
        if not chunks:
            return ""
        embeddings = self.embedder.encode(chunks, convert_to_tensor=True)
        query_emb = self.embedder.encode(query, convert_to_tensor=True)
        cos_scores = util.cos_sim(query_emb, embeddings)[0]
        top_results = torch.topk(cos_scores, k=min(self.top_k, len(chunks)))
        return "\n\n".join(chunks[idx] for idx in top_results.indices)

    def query(self, prompt: str, context: str, strict_context: bool = True) -> str:
        system = (
            "Отвечай СТРОГО по контексту. Если информации нет — скажи 'Я не знаю'."
            if strict_context else
            "Используй контекст как основу, но можешь дополнить своими знаниями."
        )
        user_msg = f"КОНТЕКСТ:\n{context}\n\nВОПРОС:\n{prompt}"
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user_msg}]
        full = system + "\n" + user_msg
        if count_tokens(self.tokenizer, full) > self.max_tokens - 1024:
            fallback = f"[ИНСТРУКЦИЯ] {'Только по контексту' if strict_context else 'Можно дополнять'}\nКОНТЕКСТ:\n{context[:2000]}...\nВОПРОС:\n{prompt}"
            messages = [{"role": "user", "content": fallback}]
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.3 if strict_context else 0.7,
            max_tokens=1024
        )
        return response.choices[0].message.content

    def query_from_text(self, prompt: str, source_text: str, strict_context: bool = True) -> str:
        context = self._prepare_context(source_text, prompt)
        return self.query(prompt, context, strict_context)

    def query_from_file(self, prompt: str, file_path: str, strict_context: bool = True) -> str:
        source_text = self._load_text_from_file(file_path)
        return self.query_from_text(prompt, source_text, strict_context)