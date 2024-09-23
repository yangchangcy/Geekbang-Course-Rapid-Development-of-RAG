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
) # 从 langchain_community.document_loaders 模块中导入各种文档加载器类

from langchain_text_splitters import RecursiveCharacterTextSplitter # 文档拆分chunk
from sentence_transformers import SentenceTransformer # 加载和使用Embedding模型
import faiss # Faiss向量库
import numpy as np # 处理嵌入向量数据，用于Faiss向量检索
import dashscope #调用Qwen大模型
from http import HTTPStatus #检查与Qwen模型HTTP请求状态

import os # 引入操作系统库，后续配置环境变量与获得当前文件路径使用
os.environ["TOKENIZERS_PARALLELISM"] = "false" # 不使用分词并行化操作, 避免多线程或多进程环境中运行多个模型引发冲突或死锁

# 设置Qwen系列具体模型及对应的调用API密钥，从阿里云百炼大模型服务平台获得
qwen_model = "qwen-turbo"
qwen_api_key = "your_api_key"

def load_document(file_path):
    """
    解析多种文档格式的文件，返回文档内容字符串
    :param file_path: 文档文件路径
    :return: 返回文档内容的字符串
    """

    # 定义文档解析加载器字典，根据文档类型选择对应的文档解析加载器类和输入参数
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

    ext = os.path.splitext(file_path)[1]  # 获取文件扩展名，确定文档类型
    loader_tuple = DOCUMENT_LOADER_MAPPING.get(ext)  # 获取文档对应的文档解析加载器类和参数元组

    if loader_tuple: # 判断文档格式是否在加载器支持范围
        loader_class, loader_args = loader_tuple  # 解包元组，获取文档解析加载器类和参数
        loader = loader_class(file_path, **loader_args)  # 创建文档解析加载器实例，并传入文档文件路径
        documents = loader.load()  # 加载文档
        content = "\n".join([doc.page_content for doc in documents])  # 多页文档内容组合为字符串
        print(f"文档 {file_path} 的部分内容为: {content[:100]}...")  # 仅用来展示文档内容的前100个字符
        return content  # 返回文档内容的字符串

    print(file_path+f"，不支持的文档类型: '{ext}'")
    return ""

def load_embedding_model():
    """
    加载bge-small-zh-v1.5模型
    :return: 返回加载的bge-small-zh-v1.5模型
    """
    print(f"加载Embedding模型中")
    # SentenceTransformer读取绝对路径下的bge-small-zh-v1.5模型，非下载
    embedding_model = SentenceTransformer(os.path.abspath('rag_app/bge-small-zh-v1.5'))
    print(f"bge-small-zh-v1.5模型最大输入长度: {embedding_model.max_seq_length}") 
    return embedding_model


def indexing_process(folder_path, embedding_model):
    """
    索引流程：加载文件夹中的所有文档文件，并将其内容分割成文档块，计算这些小块的嵌入向量并将其存储在Faiss向量数据库中。
    :param folder_path: 文档文件夹路径
    :param embedding_model: 预加载的嵌入模型
    :return: 返回Faiss嵌入向量索引和分割后的文本块原始内容列表
    """
    
    # 初始化空的chunks列表，用于存储所有文档文件的文本块
    all_chunks = []

    # 遍历文件夹中的所有文档文件
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        
        # 检查是否为文件
        if os.path.isfile(file_path):
            # 解析文档文件，获得文档字符串内容
            document_text = load_document(file_path)
            print(f"文档 {filename} 的总字符数: {len(document_text)}")
            
            # 配置RecursiveCharacterTextSplitter分割文本块库参数，每个文本块的大小为512字符（非token），相邻文本块之间的重叠128字符（非token）
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=512, chunk_overlap=128
            )
            
            # 将文档文本分割成文本块Chunk
            chunks = text_splitter.split_text(document_text)
            print(f"文档 {filename} 分割的文本Chunk数量: {len(chunks)}")
            
            # 将分割的文本块添加到总chunks列表中
            all_chunks.extend(chunks)

    # 文本块转化为嵌入向量列表，normalize_embeddings表示对嵌入向量进行归一化，用于准确计算相似度
    embeddings = []
    for chunk in all_chunks:
        embedding = embedding_model.encode(chunk, normalize_embeddings=True)
        embeddings.append(embedding)

    print("所有文本块Chunk转化为嵌入向量完成")

    # 将嵌入向量列表转化为numpy数组，FAISS索引操作需要numpy数组输入
    embeddings_np = np.array(embeddings)

    # 获取嵌入向量的维度（每个向量的长度）
    dimension = embeddings_np.shape[1]

    # 使用余弦相似度创建FAISS索引
    index = faiss.IndexFlatIP(dimension)
    # 将所有的嵌入向量添加到FAISS索引中，后续可以用来进行相似性检索
    index.add(embeddings_np)

    print("索引过程完成.")

    return index, all_chunks

def retrieval_process(query, index, chunks, embedding_model, top_k=3):
    """
    检索流程：将用户查询Query转化为嵌入向量，并在Faiss索引中检索最相似的前k个文本块。
    :param query: 用户查询语句
    :param index: 已建立的Faiss向量索引
    :param chunks: 原始文本块内容列表
    :param embedding_model: 预加载的嵌入模型
    :param top_k: 返回最相似的前K个结果
    :return: 返回最相似的文本块及其相似度得分
    """
    # 将查询转化为嵌入向量，normalize_embeddings表示对嵌入向量进行归一化
    query_embedding = embedding_model.encode(query, normalize_embeddings=True)
    # 将嵌入向量转化为numpy数组，Faiss索引操作需要numpy数组输入
    query_embedding = np.array([query_embedding])

    # 在 Faiss 索引中使用 query_embedding 进行搜索，检索出最相似的前 top_k 个结果。
    # 返回查询向量与每个返回结果之间的相似度得分（在使用余弦相似度时，值越大越相似）排名列表distances，最相似的 top_k 个文本块在原始 chunks 列表中的索引indices。
    distances, indices = index.search(query_embedding, top_k)

    print(f"查询语句: {query}")
    print(f"最相似的前{top_k}个文本块:")

    # 输出查询出的top_k个文本块及其相似度得分
    results = []
    for i in range(top_k):
        # 获取相似文本块的原始内容
        result_chunk = chunks[indices[0][i]]
        print(f"文本块 {i}:\n{result_chunk}") 

        # 获取相似文本块的相似度得分
        result_distance = distances[0][i]
        print(f"相似度得分: {result_distance}\n")

        # 将相似文本块存储在结果列表中
        results.append(result_chunk)

    print("检索过程完成.")
    return results

def generate_process(query, chunks):
    """
    生成流程：调用Qwen大模型云端API，根据查询和文本块生成最终回复。
    :param query: 用户查询语句
    :param chunks: 从检索过程中获得的相关文本块上下文chunks
    :return: 返回生成的响应内容
    """
    # 设置Qwen系列具体模型及对应的调用API密钥，从阿里云大模型服务平台百炼获得
    llm_model = qwen_model
    dashscope.api_key = qwen_api_key

    # 构建参考文档内容，格式为“参考文档1: \n 参考文档2: \n ...”等
    context = ""
    for i, chunk in enumerate(chunks):
        context += f"参考文档{i+1}: \n{chunk}\n\n"

    # 构建生成模型所需的Prompt，包含用户查询和检索到的上下文
    prompt = f"根据参考文档回答问题：{query}\n\n{context}"
    print(f"生成模型的Prompt: {prompt}")

    # 准备请求消息，将prompt作为输入
    messages = [{'role': 'user', 'content': prompt}]

    # 调用大模型API云服务生成响应
    try:
        responses = dashscope.Generation.call(
            model = llm_model,
            messages=messages,
            result_format='message',  # 设置返回格式为"message"
            stream=True,              # 启用流式输出
            incremental_output=True   # 获取流式增量输出
        )
        # 初始化变量以存储生成的响应内容
        generated_response = ""
        print("生成过程开始:")
        # 逐步获取和处理模型的增量输出
        for response in responses:
            if response.status_code == HTTPStatus.OK:
                content = response.output.choices[0]['message']['content']
                generated_response += content
                print(content, end='')  # 实时输出模型生成的内容
            else:
                print(f"请求失败: {response.status_code} - {response.message}")
                return None  # 请求失败时返回 None
        print("\n生成过程完成.")
        return generated_response
    except Exception as e:
        print(f"大模型生成过程中发生错误: {e}")
        return None

def main():
    print("RAG过程开始.")

    query="下面报告中涉及了哪几个行业的案例以及总结各自面临的挑战？"
    embedding_model = load_embedding_model()

    # 索引流程：加载文件夹中各种格式文档，分割文本块，计算嵌入向量，存储在Faiss索引中（内存）
    index, chunks = indexing_process('rag_app/data_lesson3', embedding_model)

    # 检索流程：将用户查询转化为嵌入向量，检索最相似的文本块
    retrieval_chunks = retrieval_process(query, index, chunks, embedding_model)

    # 生成流程：调用Qwen大模型生成响应
    generate_process(query, retrieval_chunks)

    print("RAG过程结束.")

if __name__ == "__main__":
    main()