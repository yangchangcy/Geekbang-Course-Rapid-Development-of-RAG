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

import chromadb # 引入Chroma向量数据库
import uuid # 生成唯一ID
import shutil # 文件操作模块，为了避免既往数据的干扰，在每次启动时清空 ChromaDB 存储目录中的文件

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
        print(f"文档 {file_path} 的部分内容为: {content[:100]}...")
        return content

    print(f"不支持的文档类型: '{ext}'")
    return ""

def load_embedding_model(model_path='rag_app/bge-small-zh-v1.5'):
    print("加载Embedding模型中")
    embedding_model = SentenceTransformer(os.path.abspath(model_path))
    print(f"bge-small-zh-v1.5模型最大输入长度: {embedding_model.max_seq_length}")
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
                # 生成每个文本块对应的唯一ID
                all_ids.extend([str(uuid.uuid4()) for _ in range(len(chunks))])

    embeddings = [embedding_model.encode(chunk, normalize_embeddings=True).tolist() for chunk in all_chunks]

    # 将文本块的ID、嵌入向量和原始文本块内容添加到ChromaDB的collection中
    collection.add(ids=all_ids, embeddings=embeddings, documents=all_chunks)
    print("嵌入生成完成，向量数据库存储完成.")
    print("索引过程完成.")
    print("********************************************************")

def retrieval_process(query, collection, embedding_model=None, top_k=6):
    query_embedding = embedding_model.encode(query, normalize_embeddings=True).tolist()

    # 使用向量数据库检索与query最相似的top_k个文本块
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)

    print(f"查询语句: {query}")
    print(f"最相似的前{top_k}个文本块:")

    retrieved_chunks = []
    # 打印检索到的文本块ID、相似度和文本块信息
    for doc_id, doc, score in zip(results['ids'][0], results['documents'][0], results['distances'][0]):
        print(f"文本块ID: {doc_id}")
        print(f"相似度: {score}")
        print(f"文本块信息:\n{doc}\n")
        retrieved_chunks.append(doc)

    print("检索过程完成.")
    print("********************************************************")
    return retrieved_chunks

def generate_process(query, chunks):
    llm_model = QWEN_MODEL
    dashscope.api_key = QWEN_API_KEY

    context = ""
    for i, chunk in enumerate(chunks):
        context += f"参考文档{i+1}: \n{chunk}\n\n"

    prompt = f"根据参考文档回答问题：{query}\n\n{context}"
    print(f"生成模型的Prompt: {prompt}")

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

    # 为了避免既往数据的干扰，在每次启动时清空 ChromaDB 存储目录中的文件
    chroma_db_path = os.path.abspath("rag_app/chroma_db")
    if os.path.exists(chroma_db_path):
        shutil.rmtree(chroma_db_path)

    # 创建ChromaDB本地存储实例和collection
    client = chromadb.PersistentClient(chroma_db_path)
    collection = client.get_or_create_collection(name="documents") 
    embedding_model = load_embedding_model()

    indexing_process('rag_app/data_lesson5', embedding_model, collection)
    query = "下面报告中涉及了哪几个行业的案例以及总结各自面临的挑战？"
    retrieval_chunks = retrieval_process(query, collection, embedding_model)
    generate_process(query, retrieval_chunks)
    print("RAG过程结束.")

if __name__ == "__main__":
    main()