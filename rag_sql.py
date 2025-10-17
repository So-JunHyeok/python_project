import os, yaml, json, re, pymysql
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.vectorstores import FAISS

class SQLRAG:
    def __init__(self, api_key, schema_path):
        self.api_key = api_key
        with open(schema_path, "r", encoding="utf-8") as f:
            schema_data = yaml.safe_load(f)

        schema_texts = []
        for table in schema_data.get("tables", []):
            tbl_name = table["table_name"]
            schema_texts.append(f"Table: {tbl_name} - {table['description']}")
            for col in table["columns"]:
                schema_texts.append(f"{tbl_name}.{col['name']} → {col['description']}")
        schema_text = "\n".join(schema_texts)

        splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
        docs = splitter.split_text(schema_text)

        embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=api_key)
        self.vectorstore = FAISS.from_texts(docs, embeddings)
        self.llm = ChatOpenAI(model="gpt-4o-mini", api_key=api_key)

    def clean_sql(self, sql_query: str) -> str:
        cleaned = re.sub(r"```[a-zA-Z]*", "", sql_query)
        cleaned = re.sub(r"```", "", cleaned)
        cleaned = re.sub(r"--.*", "", cleaned)
        cleaned = re.sub(r"/\*.*?\*/", "", cleaned, flags=re.S)
        return cleaned.strip().split(";")[0]

    def generate_sql(self, question: str):
        results = self.vectorstore.similarity_search(question, k=5)
        schema_text = "\n".join([r.page_content for r in results])

        prompt_template = """
너는 SQL 전문가다. 아래는 테이블 스키마 정보다:

{schema}

규칙:
1. 반드시 **SQL 쿼리만** 출력한다.
2. 설명, 주석, 코드블록(```sql 등)은 절대 포함하지 않는다.
3. 세미콜론(;)은 붙이지 않는다.
4. MySQL/MariaDB 호환 구문으로 작성한다.
5. 지역명칭에서 '시', '군', '구'는 제거해야 한다.
6. 단일 쿼리만 작성한다 (여러 쿼리 금지).

사용자 질문:
{question}
"""
        prompt = PromptTemplate(input_variables=["schema", "question"], template=prompt_template)
        sql_chain = LLMChain(llm=self.llm, prompt=prompt)
        sql_query = sql_chain.run(schema=schema_text, question=question)
        print(self.clean_sql(sql_query))
        return self.clean_sql(sql_query)

    def execute_sql(self, sql_query):
        conn = pymysql.connect(
            host="localhost", user="root", password="1234",
            database="seileng_test", charset="utf8mb4",
            cursorclass=pymysql.cursors.DictCursor
        )
        with conn.cursor() as cursor:
            cursor.execute(sql_query)
            results = cursor.fetchall()
        conn.close()
        return results

    def format_result(self, results):
        results_json = json.dumps(results, ensure_ascii=False)
        format_prompt = PromptTemplate(
            input_variables=["results"],
            template=(
                "아래 JSON 데이터를 Markdown 표로 변환하라.\n\n{results}\n\n"
                "규칙: 표 형태로만 출력, 컬럼명은 한국어로 변환."
            ),
        )
        format_chain = LLMChain(llm=self.llm, prompt=format_prompt)
        return format_chain.run(results=results_json)
