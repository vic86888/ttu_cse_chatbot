# query.py
from typing import Set

from langchain_ollama import ChatOllama                                    # 本地模型（Ollama）
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.chains import create_retrieval_chain                        # 官方新式檢索鏈
from langchain.chains.combine_documents import create_stuff_documents_chain

DB_DIR = "storage/chroma"
COLL_NAME = "campus_rag"

def build_chain():
    # 1) 本地 LLM（Ollama）：請先在系統上 `ollama pull llama3.1` 或你想用的模型
    llm = ChatOllama(model="cwchang/llama-3-taiwan-8b-instruct:latest", temperature=0)  # 也可換成 "qwen2.5:7b-instruct" 等

    # 2) 提示詞（Stuff Documents Chain 需要 {context} + {input}）
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "你是大學校務助理。只依據提供的文件內容回答問題，"
         "若無法從文件中找到答案，請清楚說明。請以繁體中文作答。\n\n"
         "{context}"),
        ("human", "{input}")
    ])

    # 3) Stuff chain：把檢索到的 Documents 塞進 {context}
    doc_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)

    # 4) 連接既有的 Chroma 向量庫 → 轉成 retriever
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        # 若要用 GPU，打開下一行
        model_kwargs={"device": "cuda"},
    )
    vectordb = Chroma(
        collection_name=COLL_NAME,
        embedding_function=embeddings,
        persist_directory=DB_DIR,
    )
    retriever = vectordb.as_retriever(search_kwargs={"k": 4})

    # 5) 官方新式檢索鏈：retriever + doc_chain
    return create_retrieval_chain(retriever, doc_chain)

def pretty_print_sources(context_docs):
    seen: Set[str] = set()
    lines = []
    for d in context_docs:
        src = d.metadata.get("source", "unknown")
        page = d.metadata.get("page")
        key = f"{src}#p{page}" if page is not None else src
        if key in seen:
            continue
        seen.add(key)
        if page is not None:
            lines.append(f"- {src}（第 {page} 頁）")
        else:
            lines.append(f"- {src}")
    return "\n".join(lines)

if __name__ == "__main__":
    chain = build_chain()
    print("💬 請輸入你的問題（Ctrl+C 結束）：")
    while True:
        try:
            q = input("> ")
            res = chain.invoke({"input": q})   # 回傳至少包含 answer 與 context
            print("\n🧠 答案：\n", res["answer"], "\n", sep="")
            print("📚 來源（檢索到的文件片段）：")
            print(pretty_print_sources(res["context"]))
            print("-" * 60)
        except KeyboardInterrupt:
            print("\n再見！")
            break
