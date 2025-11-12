import asyncio
import time
import numpy as np
import os
import torch
import gc
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_community.llms import HuggingFacePipeline
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from newspaper import Article
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from app.ml.collect_url import get_urls_for_keywords
from app.utils.user_command_logger import CommandPatternLogger, embeddings
import uuid
import threading, queue, sys
from datetime import datetime, timedelta
import re

class PrintManager:
    def __init__(self):
        self._lock = threading.Lock()
        self._awaiting_input = False
        self._current_prompt = ""

    def prompt(self, prompt_text: str) -> str:
        """Input wrapper: marks 'awaiting input' so bg logs can re-print prompt."""
        with self._lock:
            self._awaiting_input = True
            self._current_prompt = prompt_text
        try:
            # 실제 입력
            return input(prompt_text)
        finally:
            # 입력이 끝났으니 플래그 해제
            with self._lock:
                self._awaiting_input = False
                self._current_prompt = ""

    def bg(self, msg: str):
        """Background log: prints on a new line, then re-renders the prompt if needed."""
        with self._lock:
            # 프롬프트 줄을 깨먹지 않도록, 항상 줄을 바꿔 로그를 찍는다
            sys.stdout.write("\n" + msg.rstrip() + "\n")
            sys.stdout.flush()
            # 사용자가 입력 중이면 프롬프트를 다시 보여준다
            if self._awaiting_input and self._current_prompt:
                sys.stdout.write(self._current_prompt)
                sys.stdout.flush()

    def ui(self, *args, **kwargs):
        """Normal foreground print with lock."""
        with self._lock:
            print(*args, **kwargs)

pm = PrintManager()

MODEL_PATH = os.getenv("MODEL_PATH")

def _fmt_bytes(n):
    for unit in ["B", "KB", "MB", "GB"]:
        if n < 1024.0:
            return f"{n:,.1f}{unit}"
        n /= 1024.0
    return f"{n:,.1f}TB"

def _gpu_mem():
    if torch.cuda.is_available():
        return _fmt_bytes(torch.cuda.memory_allocated()), _fmt_bytes(torch.cuda.memory_reserved())
    return "N/A", "N/A"

# user id 생성
def get_or_create_user_id():
    path = os.path.expanduser("~/.user_id")
    if os.path.exists(path):
        with open(path, "r") as f:
            return f.read().strip()
    else:
        user_id = str(uuid.uuid4())
        with open(path, "w") as f:
            f.write(user_id)
        return user_id
    
# LLM 구성
torch.cuda.empty_cache()
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    local_files_only=True,
    device_map="auto",
    torch_dtype="auto"
)

generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512,
    temperature=0.3,
    repetition_penalty=1.05,
    top_k=20,
    top_p=0.9,
)

llm = HuggingFacePipeline(
    pipeline=generator,
    model_kwargs={"temperature": 0.4}
)

#VectorStore
PERSIST_DIR = os.path.expanduser("~/.eyestock/chroma/vectorstore_data")
os.makedirs(PERSIST_DIR, exist_ok=True)

vectorstore = Chroma(
    persist_directory=PERSIST_DIR,
    collection_name="my_collection",
    embedding_function=embeddings
)
retriever = vectorstore.as_retriever(k=5)

prompt = PromptTemplate.from_template(
    """당신은 *주식/종목* 관련 질문에만 답합니다.
- CPI, PPI, 금리 등 거시 지표는 *특정 종목/섹터의 주가 영향*과 직접 연결될 때만 간단히 언급하세요.
- 종목명, 티커, 공시/실적/주가/호가/거래량/리스크(상한가·하한가·대량거래 등) 중심으로 답하세요.
- 비주식 내용(순수 거시/정치/일반 경제 상식)만 있으면 "주식 관련 정보 부족"으로 간단히 안내하고 끝냅니다.
- 출처 문서의 내용만 근거로 사용하세요.
- 한국어로 한 문단에 2~3 개의 완결된 문장으로만 답하세요.
- 말머리글·제목·번호·불릿을 절대 쓰지 마세요.

###패턴:
{patterns}

###문서:
{documents}

###질문:
{question}

###답변:"""
)

# RAG 체인
rag_chain = prompt | llm | StrOutputParser()

# # Command Logger
# logger = CommandPatternLogger(pattern_logs, embeddings)

class BackgroundIndexer:
    def __init__(self, default_days=14):
        self.q = queue.Queue()
        self.stop_flag = False
        self.default_days = default_days
        self.seen_hashes = set()
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def submit(self, keywords, days=None):
        # 부분 선택만 있어도 바로 수집 들어가도록 큐에 적재
        if not keywords:
            return
        self.q.put({"keywords": list(dict.fromkeys(keywords)), "days": days or self.default_days})

    def shutdown(self):
        self.stop_flag = True
        self.q.put(None)
        self.thread.join(timeout=5)

    def _within_days(self, article, days):
        # 날짜 필터(가능할 때만). publish_date 없으면 통과.
        try:
            if article.publish_date:
                # naive datetime인 경우 대비
                pub = article.publish_date
                if isinstance(pub, str):
                    return True
                cutoff = datetime.now() - timedelta(days=days)
                return pub >= cutoff
        except Exception:
            pass
        return True

    def _worker(self):
        pm.bg("[BG] 백그라운드 인덱서 시작")
        while not self.stop_flag:
            task = self.q.get()
            if task is None:
                break
            keywords, days = task["keywords"], task["days"]
            try:
                urls = get_urls_for_keywords(keywords, max_per_query=200)
                pm.bg(f"[BG] 수집 시작 | 키워드={keywords} | 기간={days}일 | URL={len(urls)}개")
                total_added = 0
                for url in urls:
                    try:
                        article = Article(url)
                        article.download(); article.parse()
                        text = article.text.strip()
                        if not text or len(text) < 300 or "무단전재" in text:
                            continue
                        if not self._within_days(article, days):
                            continue
                        t_hash = hash(text[:1000])
                        if t_hash in self.seen_hashes:
                            continue
                        self.seen_hashes.add(t_hash)

                        splits = RecursiveCharacterTextSplitter(
                            chunk_size=500, chunk_overlap=0
                        ).split_text(text)
                        if splits:
                            vectorstore.add_texts(
                                splits,
                                metadatas=[{"source": url} for _ in splits]
                            )
                            total_added += len(splits)
                    except Exception as e:
                        pm.bg(f"[BG][오류] {type(e).__name__}: {e}")
                pm.bg(f"[BG] 수집 완료 | 키워드={keywords} | 추가 청크={total_added}")
            finally:
                self.q.task_done()
        pm.bg("[BG] 백그라운드 인덱서 종료")

#  RAG 세션 클래스
class RAGSession:
    def __init__(self):
        self.preferences = None
        self.has_index_built = False
        self.previous_question = None
        self.previous_embedding = None

    def is_similar_to_previous(self, question, threshold=0.9):
        if not self.previous_question:
            return False
        query_emb = embeddings.embed_query(question)
        sim = np.dot(self.previous_embedding, query_emb) / (
            np.linalg.norm(self.previous_embedding) * np.linalg.norm(query_emb)
        )
        print(f"[유사도 비교] '{self.previous_question}' vs '{question}' -> {sim:.2f}")
        return sim >= threshold

    def update_previous(self, question):
        self.previous_question = question
        self.previous_embedding = embeddings.embed_query(question)
        print(f"[이전 질문 갱신] '{question}'")

    async def collect_data_for_duration(self, question, duration_seconds):
        print(f"[수집] '{question}' → 뉴스 수집 시작 (최대 {duration_seconds}초)")
        urls = get_urls_from_question(question)
        start = time.time()
        total_added = 0
        seen_hashes = set()

        for url in urls:
            if time.time() - start > duration_seconds:
                print(f"[중단] {duration_seconds}초 초과")
                break
            try:
                article = Article(url)
                article.download()
                article.parse()
                text = article.text.strip()

                if len(text) < 300 or "무단전재" in text:
                    continue

                text_hash = hash(text[:1000])
                if text_hash in seen_hashes:
                    continue
                seen_hashes.add(text_hash)

                splits = RecursiveCharacterTextSplitter(
                    chunk_size=500, chunk_overlap=0
                ).split_text(text)

                if splits:
                    vectorstore.add_texts(
                        splits,
                        metadatas=[{"source": url} for _ in splits]
                    )
                    total_added += len(splits)
                    print(f"[추가] {len(splits)} 청크")
            except Exception as e:
                print(f"[오류] 기사 처리 실패: {e}")
            await asyncio.sleep(0)

        print(f"[수집 종료] 총 청크: {total_added}")
        

def _multi_select(prompt_text, options):
    pm.ui(prompt_text)
    for i, opt in enumerate(options, 1):
        pm.ui(f"{i}. {opt}")
    while True:
        raw = pm.prompt("번호를 쉼표로 선택 (예: 1,3,5 / 건너뛰기=엔터): ").strip()
        if not raw:  # 엔터 입력 → 선택 안 함
            return []
        
        idxs = []
        valid = True
        for tok in raw.split(","):
            tok = tok.strip()
            if not tok.isdigit():
                print(f"[오류] '{tok}'는 올바른 번호가 아닙니다. 숫자만 입력하세요.")
                valid = False
                break
            i = int(tok)
            if not (1 <= i <= len(options)):
                print(f"[오류] {i}번은 선택지 범위를 벗어났습니다.")
                valid = False
                break
            idxs.append(i - 1)
        
        if valid:
            return [options[i] for i in idxs]

def ask_user_preferences(bg: BackgroundIndexer):
    TOPICS = ["반도체","AI","2차전지","디스플레이","바이오","자동차","로봇","클라우드","국방","핀테크","에너지","플랫폼"]
    EVENTS  = ["실적발표","공시","M&A","신규수주","신제품","규제","리콜","리스트럭처링","감사보고서","자사주","배당","증자/감자"]
    MARKETS = ["코스피","코스닥","나스닥","다우","S&P500","환율","금리","원자재"]
    RISK    = ["변동성 급증","공매도","대량거래","상한가/하한가","갭상승/갭하락"]

    print("\n[관심 키워드 선택]")
    selected = []

    sel_topics = _multi_select("관심 산업/섹터를 선택하세요:", TOPICS)
    selected += sel_topics
    bg.submit(selected)  # ① 첫 선택 직후 즉시 부분 수집 시작

    sel_events = _multi_select("관심 이슈 유형을 선택하세요:", EVENTS)
    selected += sel_events
    bg.submit(selected)  # ② 누적 선택으로 즉시 수집

    sel_markets = _multi_select("시장/거시 키워드를 선택하세요:", MARKETS)
    selected += sel_markets
    bg.submit(selected)  # ③ 즉시 수집

    sel_risk = _multi_select("리스크/시그널 키워드를 선택하세요:", RISK)
    selected += sel_risk
    bg.submit(selected)  # ④ 즉시 수집

    custom = input("추가 키워드(종목/티커/자유어, 쉼표 구분, 생략 가능): ").strip()
    custom_keys = [t.strip() for t in custom.split(",") if t.strip()] if custom else []
    selected = list(dict.fromkeys(selected + custom_keys))
    bg.submit(selected)  # ⑤ 커스텀 추가 직후 수집

    # 날짜 입력(유효성 검사 루프)
    while True:
        days_raw = input("최근 몇 일치 뉴스를 인덱싱할까요? (기본 14): ").strip()
        if not days_raw:
            days = 14
            break
        if days_raw.isdigit() and int(days_raw) > 0:
            days = int(days_raw)
            break
        print("[오류] 숫자만 입력하거나 엔터로 기본값(14일)을 선택하세요.")

    # 최종 days를 반영해 한 번 더 제출(앞선 배치들은 default_days로 수집됨)
    bg.default_days = days
    bg.submit(selected, days=days)  # ⑥ 최종 튜닝

    return {"selected_keywords": selected, "days": days}


async def build_index_from_preferences(session, prefs):
    from collect_url import get_urls_from_preferences
    session.preferences = prefs
    urls = get_urls_from_preferences(prefs)

    total = len(urls)
    print(f"[인덱싱 시작] 대상 URL: {total}개 | 키워드: {prefs.get('selected_keywords', [])} | 기간: {prefs.get('days')}일")
    start = time.time()
    seen_hashes = set()
    total_added = 0
    processed = 0
    skipped_short = 0
    skipped_dup = 0
    skipped_flag = 0  # '무단전재' 등

    for idx, url in enumerate(urls, 1):
        processed += 1
        elapsed = time.time() - start
        eta = (elapsed / processed) * (total - processed) if processed else 0
        alloc, reserv = _gpu_mem()
        print(f"\r[{idx}/{total}] 파싱 중: {url} | 경과 {elapsed:,.1f}s | ETA {eta:,.1f}s | GPU alloc/resv {alloc}/{reserv}     ", end="", flush=True)
        try:
            article = Article(url)
            article.download(); article.parse()
            text = article.text.strip()

            if not text or len(text) < 300:
                skipped_short += 1
                continue
            if "무단전재" in text:
                skipped_flag += 1
                continue

            text_hash = hash(text[:1000])
            if text_hash in seen_hashes:
                skipped_dup += 1
                continue
            seen_hashes.add(text_hash)

            splits = RecursiveCharacterTextSplitter(
                chunk_size=500, chunk_overlap=0
            ).split_text(text)

            if splits:
                vectorstore.add_texts(
                    splits,
                    metadatas=[{"source": url} for _ in splits]
                )
                total_added += len(splits)
                print(f"\n[추가] {len(splits)} 청크 | 누적 {total_added} 청크")
        except Exception as e:
            print(f"\n[오류] {type(e).__name__}: {e}")
        await asyncio.sleep(0)

    print("\n[인덱싱 완료]"
          f"\n - 총 URL: {total}"
          f"\n - 처리됨: {processed}"
          f"\n - 추가된 청크: {total_added}"
          f"\n - 스킵(짧음): {skipped_short}"
          f"\n - 스킵(중복): {skipped_dup}"
          f"\n - 스킵(제외어): {skipped_flag}"
          f"\n - 총 소요: {time.time()-start:,.1f}s")
    session.has_index_built = True

async def main():
    user_id = get_or_create_user_id()
    logger = CommandPatternLogger(user_id=user_id)
    session = RAGSession()

    bg = BackgroundIndexer(default_days=14)  # 백그라운드 인덱서 시작
    try:
        # 취향 설문 (중간중간 bg.submit으로 수집 ‘즉시’ 시작)
        prefs = ask_user_preferences(bg)

        # 이후 질의응답 루프(필요 시 '설정변경'으로 다시 설문/수집)
        while True:
            question = pm.prompt("\n질문 입력 (설정변경=preferences, 종료=e): ").strip()
            if question.lower() == "e":
                break
            if question in ("preferences","설정변경"):
                prefs = ask_user_preferences(bg)
                continue

            print("[로그] 질문 기록")
            await logger.process_command(question)

            print("[로그] 패턴 검색 및 문서 검색")
            similar_patterns = logger.retrieve_patterns(question=question, k=5)
            pattern_context = "\n".join([f"- {doc.page_content}" for doc in similar_patterns])

            docs = retriever.invoke(question)
            def is_stock_chunk(text: str) -> bool:
                return any(k in text for k in [
                    "주가","공시","실적","거래량","상한가","하한가","배당",
                    "증자","감자","자사주","리포트","목표가","투자의견",
                    "코스피","코스닥","나스닥","m&a","수주","신제품"
                ])
            filtered_docs = [d for d in docs if d.page_content and is_stock_chunk(d.page_content)]    

            doc_context = "\n\n".join(d.page_content for d in filtered_docs[:30])

            prompt_input = {"patterns": pattern_context, "documents": doc_context, "question": question}
            
            def _postprocess_chatty(text: str) -> str:
                t = re.sub(r"(?m)^\s*([\-–•\*\d]+\s*[.)]?)\s*", "", text)  # 불릿/번호 제거
                t = re.sub(r"\s*\n+\s*", " ", t).strip()                   # 줄바꿈 합치기
                sents = re.split(r"(?<=[.!?。])\s+", t)                    # 문장 분리
                t = " ".join(sents[:3])                                    # 최대 3문장
                return t
            
            raw_output = rag_chain.invoke(prompt_input)
            answer = raw_output.split("###답변:")[-1].strip() if "###답변:" in raw_output else raw_output.strip()
            answer = _postprocess_chatty(answer)
            print(f"[답변]\n{answer}\n")

            gc.collect(); torch.cuda.empty_cache()
    finally:
        bg.shutdown()  # 종료 시 워커 안전 종료

# 시작
if __name__ == "__main__":
    asyncio.run(main())
