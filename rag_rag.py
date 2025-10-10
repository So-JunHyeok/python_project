import faiss, pickle, numpy as np, os
from openai import OpenAI

class DocumentRAG:
    def __init__(self, api_key, index_file, meta_file):
        self.client = OpenAI(api_key=api_key)
        self.index = faiss.read_index(index_file)
        with open(meta_file, "rb") as f:
            self.metadatas = pickle.load(f)

    def get_embedding(self, text, model="text-embedding-3-small"):
        resp = self.client.embeddings.create(model=model, input=text)
        return np.array(resp.data[0].embedding, dtype="float32")

    def search(self, query, k=3):
        q_vec = self.get_embedding(query).reshape(1, -1)
        D, I = self.index.search(q_vec, k=50)
        results = []
        for idx, score in zip(I[0], D[0]):
            if idx == -1:
                continue
            meta = self.metadatas[idx]
            results.append({
                "score": float(score),
                "content": meta.get("content", ""),
                "type": meta.get("type"),
                "page": meta.get("page"),
                "doc_id": meta.get("doc_id")
            })
            if len(results) >= k:
                break
        return results

    def generate_answer(self, query, contexts):
        context_text = "\n\n".join(
            [f"[{c['doc_id']} - {c['type']} p{c['page']}]\n{c.get('content','')}" for c in contexts]
        )
        prompt = f"""
질문: {query}

아래 Context만 사용해서 답변하세요.
출력 규칙:
1. 문장을 절대 요약하거나 재작성하지 말 것.
2. 문서의 원문 문장을 그대로 발췌할 것.
3. 각 문장은 '-' bullet 형식으로만 출력할 것.
4. 각 bullet 끝에는 [doc_id, pX] 출처를 붙일 것.

Context:
{context_text}
"""
        resp = self.client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0,
            messages=[
                {"role": "system", "content": "문서 기반 QA 보조자입니다."},
                {"role": "user", "content": prompt}
            ]
        )
        return resp.choices[0].message.content.strip()
