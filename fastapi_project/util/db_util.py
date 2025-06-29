import psycopg2
from psycopg2 import Error
import re 
import os
import chromadb
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, DocumentSummaryIndex
from llama_index.core import PromptTemplate, get_response_synthesizer, load_index_from_storage
from llama_index.core import Document, StorageContext
from llama_index.core.settings import Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.deepseek import DeepSeek
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.response_synthesizers import ResponseMode
from .chat_util import initialize_llamaindex
from ..settings import settings

##获取连接对象
def connect_to_postgres():
    try:
        # 连接到PostgreSQL数据库
        connection = psycopg2.connect(
            user=settings.USERS,
            password=settings.PASSWORD,
            host="localhost",
            port="5432",
            database=settings.DATABASE
        )
        return connection
    except Error as e:
        print(f"连接数据库时出错: {e}")
        return None

#执行查询
def execute_query(query, params=None):
    try:
        connection = connect_to_postgres()
        if connection is None:
            return
        
        cursor = connection.cursor()
        
        # 执行查询
        if params:
            cursor.execute(query, params)
        else:
            cursor.execute(query)
            
        # 如果是SELECT查询，获取结果
        if query.strip().upper().startswith('SELECT'):
            results = cursor.fetchall()
            # 获取列名
            column_names = [desc[0] for desc in cursor.description] if cursor.description else []
            # print("列名:", column_names)
            # print("查询结果:", results)
            return results
        else:
            # 对于INSERT、UPDATE、DELETE操作，获取受影响的行数
            affected_rows = cursor.rowcount
            print(f"操作影响的行数: {affected_rows}")
            
        # 提交事务
        connection.commit()
        
    except Error as e:
        print(f"执行查询时出错: {e}")
    finally:
        if connection:
            cursor.close()
            connection.close()
            print("数据库连接已关闭")

#创建article表
def create_article_table():
    query = """
    CREATE TABLE IF NOT EXISTS article (
        id SERIAL,
        title TEXT,
        body TEXT,
        tags text[],
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """
    execute_query(query)
    return None

#创建message表
def create_message_table():
    query = """
DROP TABLE IF EXISTS message;
CREATE TABLE IF NOT EXISTS message (
    user_id TEXT,
    session_id TEXT,
    history TEXT,
    update_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    abstract TEXT DEFAULT '这是摘要'
);"""
    execute_query(query)
    return None

##将指定文件夹中的文件写入pgsql的article表
def write_to_article(txt_dir):
    import os
    from datetime import datetime

    # 获取txt_split_file目录下所有txt文件
    txt_files = [f for f in os.listdir(txt_dir) if f.endswith('.txt')]

    for txt_file in txt_files:
        # 构建完整的文件路径
        file_path = os.path.join(txt_dir, txt_file)
        
        # 读取文件内容
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 准备SQL插入语句
        insert_query = """
        INSERT INTO article (title, body, tags, updated_at)
        VALUES (%s, %s, %s, %s)
        """
        
        # 准备参数
        params = (
            txt_file,  # title
            content,   # body
            [],        # tags (空数组)
            datetime.now()  # updated_at
        )
        
        # 执行插入
        execute_query(insert_query, params)
    return None

##随机生成三条message信息到postgresql数据库的message表
import uuid
import json
def generate_message_info():
    # 生成3个不同的user_id
    user_ids = [str(uuid.uuid4()) for _ in range(3)]
    # 为每个user_id生成3个不同的session_id并创建对应的历史记录
    for user_id in user_ids:
        # 为每个用户生成3个session_id
        session_ids = ['session_'+str(uuid.uuid4()) for _ in range(3)]
        # 为每个session创建历史记录
        for session_id in session_ids:
            chat_history = [
                {"role": "assistant", "content": "你好"},
                {"role": "user", "content": "邹韬奋是谁？"},
                {"role": "assistant", "content": "他是一个伟大的作家"},
                {"role": "user", "content": "你吃了么"},
            ]
            
            # 将历史记录转换为JSON字符串
            history_json = json.dumps(chat_history, ensure_ascii=False)
            
            # 插入数据库
            insert_query = f"""
            INSERT INTO message (user_id, session_id, history)
            VALUES ('{user_id}', '{session_id}', '{history_json}');
            """
            execute_query(insert_query)
    return None




## 从原始文档中生成两级索引，一层是summaryIndex
##创建摘要索引，自定义摘要内容
def generate_summary_index(documents):
    #初始化 Chroma 客户端
    chroma_client = chromadb.Client()
    chroma_collection = chroma_client.get_or_create_collection("text_doc1")
    # 创建向量存储orm
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    summary_tmpl = PromptTemplate(
    "请用1-2句话总结以下文档，并提出1-2个该文档能回答的用户问题：\n"
    "{context_str}\n"
    "—— 摘要：")
    # LlamaIndex ≥0.10.x 把 summary_template 收进 response_synthesizer
    resp_synth = get_response_synthesizer(
        response_mode=ResponseMode.TREE_SUMMARIZE,
        summary_template=summary_tmpl
    )
    # 构建 DocumentSummaryIndex
    
    doc_sum_index = DocumentSummaryIndex.from_documents(
        documents,
        response_synthesizer=resp_synth,
        vector_store=vector_store
    )
    return doc_sum_index


def generate_simple_index(documents):
    # 按换行符分割，并禁用自动句子分割
    splitter = SentenceSplitter(
        separator="\n\n",  # 强制按换行符切分
        chunk_size=500,  # 可选：防止单个node过长
        chunk_overlap=0,  # 无重叠
        paragraph_separator="\n\n",  # 明确段落分隔符
        secondary_chunking_regex="\n\n",  # 二级分割同样用换行符
    )

    # 分割并保留doc_id
    nodes = []
    for doc in documents:
        # 对当前doc生成nodes
        doc_nodes = splitter.get_nodes_from_documents([doc])
        # 为每个node添加原doc_id到metadata
        for node in doc_nodes:
            node.metadata["doc_id"] = doc.doc_id
        nodes.extend(doc_nodes)

    # 验证输出
    for node in nodes:
        print(f"Node ID: {node.node_id}, Doc ID: {node.metadata['doc_id']}, Text: {node.text[:20]}...")

    # 创建索引（新版无需ServiceContext）
    index = VectorStoreIndex(nodes)
    return index

##根据用户提问，使用摘要索引获得相关文档
def get_docs_from_summaryindex(doc_sum_index,query):
    retriever = doc_sum_index.as_retriever(similarity_top_k=3)
    # 手动进行检索（结果为 NodeWithScore 对象）
    retrieved_nodes = retriever.retrieve(query)
     # 使用集合去重
    seen_doc_ids = set()
    result = []
    for node in retrieved_nodes:
        ref_doc_id = node.node.ref_doc_id
        # 只添加未见过的文档ID
        if ref_doc_id not in seen_doc_ids:
            seen_doc_ids.add(ref_doc_id)
            result.append({
                'ref_doc_id': ref_doc_id,
                'score': node.score,  # 相关性分数
                'text': node.node.text[:100] + '...',  # 文本预览
            })
    
    return result

##仅仅创建两个index
def create_summary_and_simple_index():
    # 初始化LLM和嵌入模型
    llm, embed_model = initialize_llamaindex(
        deepseekapi=settings.DEEPSEEK_API  # 替换为您的DeepSeek API密钥
    )
    documents = SimpleDirectoryReader(input_dir="fastapi_project\\data\\").load_data()
    for doc in documents:
        # 将原始doc_id存储在metadata中
        doc.metadata['original_doc_id'] = doc.doc_id
    doc_sum_index = generate_summary_index(documents)
    index = generate_simple_index(documents)
    return index,doc_sum_index

##保存两个index
def save_indexes(index,doc_sum_index):
    index.storage_context.persist("fastapi_project\\store\\simpleindex")
    doc_sum_index.storage_context.persist("fastapi_project\\store\\summaryindex")

##加载两个index
def load_indexes():
    # 使用已初始化的模型实例，避免重复初始化
    # from .chat_util import _llm_instance, _embed_model_instance
    
    # # 如果模型还未初始化，则进行初始化
    # if _llm_instance is None or _embed_model_instance is None:
    #     llm, embed_model = initialize_llamaindex(
    #         deepseekapi=settings.DEEPSEEK_API
    #     )
    llm, embed_model = initialize_llamaindex(deepseekapi=settings.DEEPSEEK_API)
    # 加载VectorStoreIndex)
    storage_context = StorageContext.from_defaults(persist_dir="fastapi_project\\store\\simpleindex")
    loaded_simpleindex = load_index_from_storage(storage_context)

    # 加载DocumentSummaryIndex
    storage_context = StorageContext.from_defaults(persist_dir="fastapi_project\\store\\summaryindex")
    loaded_doc_sum_index = load_index_from_storage(storage_context)
    return loaded_simpleindex,loaded_doc_sum_index

# 创建检索器，检索器只能检索信息，不需要进一步创建搜索或对话引擎
def get_final_nodes_text(loaded_simpleindex,loaded_doc_sum_index,user_query):
    res = get_docs_from_summaryindex(loaded_doc_sum_index,query=user_query)
    selected_doc_ids = [r['ref_doc_id'] for r in res]
    retriever = loaded_simpleindex.as_retriever(similarity_top_k=5)
    retrieved_docs = retriever.retrieve(user_query)
    res_text = ''
    for nodeWithScore in retrieved_docs:
        if nodeWithScore.node.metadata['original_doc_id'] in selected_doc_ids:
            res_text += nodeWithScore.node.text + '\n\n'
    return res_text

if __name__=="__main__":
    print(settings.DEEPSEEK_API)
    print(".env文件信息成功加载！")
    # index,doc_sum_index=create_summary_and_simple_index()
    # save_indexes(index,doc_sum_index)
    # print("两个索引创建成功")
    # loaded_simpleindex,loaded_doc_sum_index=load_indexes()
    # print("两个索引加载成功")
    # res_text=get_final_nodes_text(loaded_simpleindex,loaded_doc_sum_index,user_query="章先生是谁？")
    # print(res_text)
    connect_to_postgres()
    print("数据库连接成功！")
    create_article_table()
    print("article表创建成功")
    create_message_table()
    print("message表创建成功")
    generate_message_info()
    print("message表随机生成三条信息成功")
    write_to_article(settings.ARTICLE_DIR)
    print("txt_files文件夹中的txt文件写入pgsql的article表成功!")
    load_indexes()
    print("两个索引加载成功")
