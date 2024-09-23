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

from rank_bm25 import BM25Okapi # 从 rank_bm25 库中导入 BM25Okapi 类，用于实现 BM25 算法的检索功能
import jieba # 导入 jieba 库，用于对中文文本进行分词处理

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

    # 从 Chroma collection 中提取所有文档
    all_docs = collection.get()['documents']

    # 对所有文档进行中文分词
    tokenized_corpus = [list(jieba.cut(doc)) for doc in all_docs]

    # 使用分词后的文档集合实例化 BM25Okapi，对这些文档进行 BM25 检索的准备工作
    bm25 = BM25Okapi(tokenized_corpus)
    # 对查询语句进行分词处理，将分词结果存储为列表
    tokenized_query = list(jieba.cut(query))
    # 计算查询语句与每个文档的 BM25 得分，返回每个文档的相关性分数
    bm25_scores = bm25.get_scores(tokenized_query)
    
    # 获取 BM25 检索得分最高的前 top_k 个文档的索引
    bm25_top_k_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:top_k]
    # 根据索引提取对应的文档内容
    bm25_chunks = [all_docs[i] for i in bm25_top_k_indices]

    # 打印 向量 检索结果
    print(f"查询语句: {query}")
    print(f"向量检索最相似的前 {top_k} 个文本块:")
    vector_chunks = []
    for rank, (doc_id, doc) in enumerate(zip(vector_results['ids'][0], vector_results['documents'][0])):
        print(f"向量检索排名: {rank + 1}")
        print(f"文本块ID: {doc_id}")
        print(f"文本块信息:\n{doc}\n")
        vector_chunks.append(doc)

    # 打印 BM25 检索结果
    print(f"BM25 检索最相似的前 {top_k} 个文本块:")
    for rank, doc in enumerate(bm25_chunks):
        print(f"BM25 检索排名: {rank + 1}")
        print(f"文档内容:\n{doc}\n")

    # 合并结果，将 向量 检索的结果放在前面，然后是 BM25 检索的结果
    combined_results = vector_chunks + bm25_chunks

    print("检索过程完成.")
    print("********************************************************")

    # 返回合并后的全部结果，共2*top_k个文档块
    return combined_results

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