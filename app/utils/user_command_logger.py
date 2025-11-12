import asyncio
import time
import numpy as np
from langchain_chroma import Chroma
from datetime import datetime, timedelta
import os
from sentence_transformers import SentenceTransformer
from app.ml.embed_loader import get_embedder

EMBEDDING_PATH = os.getenv("EMBEDDING_PATH")

# === Embedding 모델 ===
usr_embedder = get_embedder()


class EmbeddingWrapper:
    def __init__(self, usr_embedder):
        self.usr_embedder = usr_embedder

    def embed_query(self, text):
        return self.usr_embedder.encode([text])[0]

    def embed_documents(self, texts):
        return self.usr_embedder.encode(texts)


embeddings = EmbeddingWrapper(usr_embedder)

# === 사용자 패턴 벡터스토어 ===
USER_LOGS_ROOT = os.path.expanduser("~/.eyestock/chroma/user_logs")
os.makedirs(USER_LOGS_ROOT, exist_ok=True)


def _make_chroma_store(persist_dir: str, collection_name: str, embedding_fn):
    """
    Try new Chroma PersistentClient first (권장).
    If chromadb version is older, fall back to legacy Chroma(persist_directory=...).
    """
    try:
        import chromadb
        from chromadb.config import Settings
        client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(allow_reset=True)
        )
        return Chroma(
            client=client,
            collection_name=collection_name,
            embedding_function=embedding_fn,
        )
    except Exception:
        # 구 방식 폴백
        os.makedirs(persist_dir, exist_ok=True)
        return Chroma(
            persist_directory=persist_dir,
            collection_name=collection_name,
            embedding_function=embedding_fn,
        )


# === 사용자 패턴 로거 ===
class CommandPatternLogger:
    """
    사용자별 명령 패턴 로깅 및 유사도 기반 중복 검출
    """
    def __init__(self, user_id: str, base_dir: str = USER_LOGS_ROOT):
        self.user_id = user_id
        self.embeddings = embeddings

        user_store_dir = os.path.join(base_dir, user_id)
        os.makedirs(user_store_dir, exist_ok=True)

        self.vectorstore = _make_chroma_store(
            persist_dir=user_store_dir,
            collection_name=f"user_logs_{user_id}",
            embedding_fn=self.embeddings,
        )

        self.retriever = self.vectorstore.as_retriever(k=5)
        self.previous_commands: List[str] = []
        self.previous_embeddings = {}

        try:
            data = self.vectorstore.get()
            stored_docs = data.get("documents", []) if isinstance(data, dict) else []
            for text in stored_docs:
                if text and text.strip():
                    self.previous_commands.append(text)
                    self.previous_embeddings[text] = self.embeddings.embed_query(text)
            print(f"[초기화] 유저 '{user_id}' 질문 {len(self.previous_commands)}개 로드 완료")
        except Exception as e:
            print(f"[오류] 유저 '{user_id}' 벡터스토어 로딩 실패: {e}")

    def similarity_to_previous(self, command: str, threshold: float = 0.85):
        if not self.previous_commands:
            return False, 0.0

        if command not in self.previous_embeddings:
            self.previous_embeddings[command] = self.embeddings.embed_query(command)

        query_emb = self.previous_embeddings[command]
        prev_embs = np.array([self.previous_embeddings[c] for c in self.previous_commands])

        denom = (np.linalg.norm(prev_embs, axis=1) * np.linalg.norm(query_emb) + 1e-12)
        sim = np.dot(prev_embs, query_emb) / denom
        max_sim = float(np.max(sim))
        return max_sim >= threshold, max_sim

    def add_command(self, command: str):
        if command not in self.previous_embeddings:
            self.previous_embeddings[command] = self.embeddings.embed_query(command)
        self.previous_commands.append(command)
        if len(self.previous_commands) > 50:
            oldest = self.previous_commands.pop(0)
            self.previous_embeddings.pop(oldest, None)

    def store_command(self, command: str):
        tagged = command
        self.vectorstore.add_texts([tagged])
        self.add_command(tagged)

    async def process_command(self, command: str) -> bool:
        similar, sim_val = self.similarity_to_previous(command)
        print(f"[유사도] 이전 명령과 {sim_val:.2f}")
        if not similar:
            self.store_command(command)
            print("[저장] 새 패턴을 벡터스토어에 기록.")
        else:
            print("[스킵] 유사 패턴이므로 중복 저장 생략.")
        return similar

    def retrieve_patterns(self, question: str, k: int = 5, user_id: str | None = None):
        retriever = self.vectorstore.as_retriever(k=k)
        query = f"[{user_id}] {question}" if user_id else question
        results = retriever.invoke(query)

        print(f"[패턴 검색 결과] 총 {len(results)}건:")
        for doc in results:
            try:
                print(f"- {doc.page_content.strip()[:100]}")
            except Exception:
                pass
        return results

# __all__ 갱신: 전역 vectorstore는 제거 (충돌 방지)
__all__ = ["CommandPatternLogger", "embeddings"]

