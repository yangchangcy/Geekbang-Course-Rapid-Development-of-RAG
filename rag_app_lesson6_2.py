from langchain_community.document_loaders import (
    PDFPlumberLoader,
    TextLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredPowerPointLoader,
    UnstructuredExcelLoader,
    CSVLoader,
    UnstructuredMarkdownLoader,
    UnstructuredXMLLoader,
    UnstructuredHTMLLoader,
)

from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import os
import dashscope
from http import HTTPStatus

import chromadb
import uuid
import shutil

from rank_bm25 import BM25Okapi
import jieba

from FlagEmbedding import FlagReranker # 用于对嵌入结果进行重新排序的工具类

os.environ["TOKENIZERS_PARALLELISM"] = "false"
QWEN_MODEL = "qwen-turbo"
QWEN_API_KEY = "your_api_key"

DOCUMENT_LOADER_MAPPING = {
    ".pdf": (PDFPlumberLoader, {}),
    ".txt": (TextLoader, {"encoding": "utf8"}),
    ".doc": (UnstructuredWordDocumentLoader, {}),
    ".docx": (UnstructuredWordDocumentLoader, {}),
    ".ppt": (UnstructuredPowerPointLoader, {}),
    ".pptx": (UnstructuredPowerPointLoader, {}),
    ".xlsx": (UnstructuredExcelLoader, {}),
    ".csv": (CSVLoader, {}),
    ".md": (UnstructuredMarkdownLoader, {}),
    ".xml": (UnstructuredXMLLoader, {}),
    ".html": (UnstructuredHTMLLoader, {}),
}

def load_document(file_path):
    ext = os.path.splitext(file_path)[1]
    loader_class, loader_args = DOCUMENT_LOADER_MAPPING.get(ext, (None, None))

    if loader_class:
        loader = loader_class(file_path, **loader_args)
        documents = loader.load()
        content = "\n".join([doc.page_content for doc in documents])
        return content

    print(f"不支持的文档类型: '{ext}'")
    return ""

def load_embedding_model(model_path='rag_app/bge-small-zh-v1.5'):
    print("加载Embedding模型中")
    embedding_model = SentenceTransformer(os.path.abspath(model_path))
    print(f"bge-small-zh-v1.5模型最大输入长度: {embedding_model.max_seq_length}\n")
    return embedding_model

def reranking(query, chunks, top_k=3):
    # 初始化重排序模型，使用BAAI/bge-reranker-v2-m3
    reranker = FlagReranker('BAAI/bge-reranker-v2-m3', use_fp16=True)
    
    # 构造输入对，每个 query 与 chunk 形成一对
    input_pairs = [[query, chunk] for chunk in chunks]
    
    # 计算每个 chunk 与 query 的语义相似性得分
    scores = reranker.compute_score(input_pairs, normalize=True)
    
    print("文档块重排序得分:", scores)
    
    # 对得分进行排序并获取排名前 top_k 的 chunks
    sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    reranking_chunks = [chunks[i] for i in sorted_indices[:top_k]]
    
    # 打印前三个 score 对应的文档块
    for i in range(top_k):
        print(f"重排序文档块{i+1}: 相似度得分：{scores[sorted_indices[i]]}，文档块信息：{reranking_chunks[i]}\n")
    
    return reranking_chunks

def indexing_process(folder_path, embedding_model, collection):
    all_chunks = []
    all_ids = []

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        if os.path.isfile(file_path):
            document_text = load_document(file_path)
            if document_text:
                print(f"文档 {filename} 的总字符数: {len(document_text)}")

                text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=128)
                chunks = text_splitter.split_text(document_text)
                print(f"文档 {filename} 分割的文本Chunk数量: {len(chunks)}")

                all_chunks.extend(chunks)
                all_ids.extend([str(uuid.uuid4()) for _ in range(len(chunks))])

    embeddings = [embedding_model.encode(chunk, normalize_embeddings=True).tolist() for chunk in all_chunks]

    collection.add(ids=all_ids, embeddings=embeddings, documents=all_chunks)
    print("嵌入生成完成，向量数据库存储完成.")
    print("索引过程完成.")
    print("********************************************************")

def retrieval_process(query, collection, embedding_model=None, top_k=6):

    query_embedding = embedding_model.encode(query, normalize_embeddings=True).tolist()
    vector_results = collection.query(query_embeddings=[query_embedding], n_results=top_k)

    all_docs = collection.get()['documents']

    tokenized_corpus = [list(jieba.cut(doc)) for doc in all_docs]

    bm25 = BM25Okapi(tokenized_corpus)
    tokenized_query = list(jieba.cut(query))
    bm25_scores = bm25.get_scores(tokenized_query)
    
    bm25_top_k_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:top_k]
    bm25_chunks = [all_docs[i] for i in bm25_top_k_indices]

    print(f"查询语句: {query}")
    print(f"向量检索最相似的前 {top_k} 个文本块:")
    vector_chunks = []
    for rank, (doc_id, doc) in enumerate(zip(vector_results['ids'][0], vector_results['documents'][0])):
        print(f"向量检索排名: {rank + 1}")
        print(f"文本块ID: {doc_id}")
        print(f"文本块信息:\n{doc}\n")
        vector_chunks.append(doc)

    print(f"BM25 检索最相似的前 {top_k} 个文本块:")
    for rank, doc in enumerate(bm25_chunks):
        print(f"BM25 检索排名: {rank + 1}")
        print(f"文档内容:\n{doc}\n")

    # 使用重排序模型对检索结果进行重新排序，输出重排序后的前top_k文档块
    reranking_chunks = reranking(query,vector_chunks + bm25_chunks, top_k)

    print("检索过程完成.")
    print("********************************************************")

    # 返回重排序后的前top_k个文档块
    return reranking_chunks

def generate_process(query, chunks):
    llm_model = QWEN_MODEL
    dashscope.api_key = QWEN_API_KEY

    context = ""
    for i, chunk in enumerate(chunks):
        context += f"参考文档{i+1}: \n{chunk}\n\n"

    prompt = f"根据参考文档回答问题：{query}\n\n{context}"
    print(prompt+"\n")

    messages = [{'role': 'user', 'content': prompt}]

    try:
        responses = dashscope.Generation.call(
            model = llm_model,
            messages=messages,
            result_format='message', 
            stream=True,            
            incremental_output=True   
        )
        generated_response = ""
        print("生成过程开始:")
        for response in responses:
            if response.status_code == HTTPStatus.OK:
                content = response.output.choices[0]['message']['content']
                generated_response += content
                print(content, end='')
            else:
                print(f"请求失败: {response.status_code} - {response.message}")
                return None
        print("\n生成过程完成.")
        print("********************************************************")
        return generated_response
    except Exception as e:
        print(f"大模型生成过程中发生错误: {e}")
        return None

def main():
    print("RAG过程开始.")

    chroma_db_path = os.path.abspath("rag_app/chroma_db")
    if os.path.exists(chroma_db_path):
        shutil.rmtree(chroma_db_path)

    client = chromadb.PersistentClient(path=os.path.abspath(chroma_db_path))
    collection = client.get_or_create_collection(name="documents") 
    embedding_model = load_embedding_model()

    indexing_process('rag_app/data_lesson6', embedding_model, collection)
    query = "下面报告中涉及了哪几个行业的案例以及总结各自面临的挑战？"
    retrieval_chunks = retrieval_process(query, collection, embedding_model)
    generate_process(query, retrieval_chunks)
    print("RAG过程结束.")

if __name__ == "__main__":
    main()