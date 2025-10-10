from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
from rag_rag import DocumentRAG
from rag_sql import SQLRAG
from openai import OpenAI

# ✅ LLM 클라이언트
api_key = os.getenv("LLM_API_KEY", "YOUR_API_KEY")
client = OpenAI(api_key=api_key)

# ✅ 두 RAG 엔진 초기화
doc_rag = DocumentRAG(api_key, r"C:\Users\User\rag_pipeline\faiss_text.index", r"C:\Users\User\rag_pipeline\meta_text.pkl")
sql_rag = SQLRAG(api_key, r"C:\Users\User\codeTest\codeTest\schema.yaml")

# ✅ FastAPI 초기화
app = FastAPI(title="Unified RAG Server")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------
# 🔍 질의 유형 자동 분류
# ---------------------------
def detect_query_type(query: str) -> str:
    """
    규칙 기반 + LLM 보조판단 결합형 라우팅
    """
    sql_keywords = ["평균", "합계", "데이터", "통계", "수온", "기온", "조회", "년", "월별", "일별"]
    doc_keywords = ["설명", "절차", "기준", "지침", "방법", "문서", "규정"]

    # 1️⃣ 우선 규칙 기반 판단
    if any(k in query for k in sql_keywords):
        return "sql"
    if any(k in query for k in doc_keywords):
        return "doc"

    # 2️⃣ LLM 보조 판단 (애매한 경우)
    prompt = f"""
    아래 질의가 데이터베이스 조회(SQL형)인지, 문서검색(RAG형)인지 판단해줘.
    - 통계, 평균, 합계, 수치값 → SQL
    - 설명, 절차, 기준, 지침 → RAG
    출력은 'SQL' 또는 'RAG' 둘 중 하나만.
    질의: {query}
    """
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    mode = resp.choices[0].message.content.strip().upper()
    return "sql" if mode == "SQL" else "doc"

# ---------------------------
# 📡 API 요청 모델
# ---------------------------
class AskRequest(BaseModel):
    query: str

# ---------------------------
# 📡 단일 API 엔드포인트
# ---------------------------
@app.post("/ask")
def ask(req: AskRequest):
    # 🔍 질의 유형 자동 감지
    mode = detect_query_type(req.query)
    print(mode)

    if mode == "doc":
        results = doc_rag.search(req.query)
        answer = doc_rag.generate_answer(req.query, results)
        return {"mode": "doc", "query": req.query, "answer": answer, "contexts": results}

    elif mode == "sql":
        sql_query = sql_rag.generate_sql(req.query)
        data = sql_rag.execute_sql(sql_query)
        formatted = sql_rag.format_result(data)
        return {"mode": "sql", "query": req.query, "answer": formatted, "result": formatted}

    else:
        return {"error": "Invalid mode detected"}
