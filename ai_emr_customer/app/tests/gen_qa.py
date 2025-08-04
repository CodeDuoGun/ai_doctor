import requests
import tempfile
import os
import traceback
import time
import json
import math
from openai import OpenAI
from tqdm import tqdm
from app.config import config
import re
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter,SpacyTextSplitter
from langchain_community.document_loaders import (
    CSVLoader,
    EverNoteLoader,
    PyMuPDFLoader,
    TextLoader,
    UnstructuredEmailLoader,
    UnstructuredEPubLoader,
    UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader,
    UnstructuredODTLoader,
    UnstructuredPowerPointLoader,
    UnstructuredWordDocumentLoader,
)
from typing import List
from app.utils.log import logger
import argparse
import pandas as pd
qa_map = {"Q": [], "A": []}
# 配置（请在使用时替换为实际的URL和API密钥）
# base_url = 'https://oapi.tasking.ai/v1'
# api_key = 'tkFunTEaWaNt9Aity5FoAaLk1kwpiWK3'
# headers = {"Authorization": f"Bearer {api_key}"}

# OpenAI客户端配置（请在使用时替换为实际的API密钥和URL）
client = OpenAI(
    api_key="f43df693-455a-4e3e-b987-d76d7b57f4c3",
    base_url="https://ark.cn-beijing.volces.com/api/v3",
)

# Document loaders mapping
LOADER_MAPPING = {
    ".csv": (CSVLoader, {}),
    ".doc": (UnstructuredWordDocumentLoader, {}),
    ".docx": (UnstructuredWordDocumentLoader, {}),
    ".enex": (EverNoteLoader, {}),
    ".eml": (UnstructuredEmailLoader, {}),
    ".epub": (UnstructuredEPubLoader, {}),
    ".html": (UnstructuredHTMLLoader, {}),
    ".md": (UnstructuredMarkdownLoader, {}),
    ".odt": (UnstructuredODTLoader, {}),
    ".pdf": (PyMuPDFLoader, {}),
    ".ppt": (UnstructuredPowerPointLoader, {}),
    ".pptx": (UnstructuredPowerPointLoader, {}),
    ".txt": (TextLoader, {"encoding": "utf8"}),
}

# def get_completion(prompt, model="qwen25-72b"):
def get_completion(prompt, model="deepseek-v3-250324", max_retries=3, retry_delay=2):
    """获取模型的响应"""
    retries = 0
    while retries < max_retries:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": "你是一个智能文档切割专家，根据用户要求，给出有效的切割结果"},{"role": "user", "content": prompt}],
                temperature=0,
            )
            return response.choices[0].message.content
        except Exception as e:
            retries += 1
            if retries >= max_retries:
                logger.debug(f"经过 {max_retries} 次重试后，调用API仍然失败: {traceback.format_exc()}")
                return None
            time.sleep(retry_delay)
            logger.warning(f"调用API失败，正在第 {retries} 次重试...")
    return None

def packet_qa(res_map, text):
    """res_map{"Q": [], "A": []}}"""
    res = text.split("\n" )
    logger.debug(f"splited res: {res}")
    # for t in res:
    #     if not t:
    #         continue
    #     if "Q" in t:
    #         t = t.strip("Q:").strip(" ")
    #         res_map["Q"].append(t)
    #     if "A" in t:
    #         t = t.strip("A:").strip(" ")
    #         res_map["A"].append(t)
    # assert len(res_map["Q"])  == len(res_map["A"])
    logger.debug(f"res_map: {res_map}")
    return res_map


def generate_qa_pairs_with_progress(text_chunks):
    """生成问答对并显示进度"""
    global qa_map
    qa_pairs = []
    for i, chunk in enumerate(tqdm(text_chunks, desc="Processing chunks")):
        prompt = f"""基于以下给定的文本，生成一组高质量的问答对。请遵循以下指南：
        
                1. 问题部分：
                - 为同一个主题创建尽可能多的（如K个）不同表述的问题，确保问题的多样性。
                - 每个问题应考虑用户可能的多种问法并且和主题强相关，例如：
                - 直接询问（如“什么是...？”）
                - 请求确认（如“是否可以说...？”
                - 寻求解释（如“请解释一下...的含义。”）
                - 假设性问题（如“如果...会怎样？”）
                - 例子请求（如“能否举个例子说明...？”）
                - 问题应涵盖文本中的关键信息、主要概念和细节，确保不遗漏重要内容。

                2. 答案部分：
                - 提供一个全面、详细的答案，涵盖问题的所有可能角度，确保逻辑连贯。
                - 答案应直接基于给定文本，确保准确性和一致性。
                - 包含相关的细节，如日期、名称、职位等具体信息，必要时提供背景信息以增强理解。

                3. 格式：
                - 使用 "Q:" 标记问题集合的开始，所有问题应在一个段落内，问题之间用空格分隔。
                - 使用 "A:" 标记答案的开始，答案应清晰分段，便于阅读。
                - 问答对之间用两个空行分隔，以提高可读性。

                4. 内容要求：
                - 确保问答对紧密围绕文本主题，避免偏离主题。
                - 避免添加文本中未提及的信息，确保信息的真实性。
                - 如果文本信息不足以回答某个方面，则扔掉这组QA。
                - 确保QA的数量相同。
                - 如果多个Q对应一个A，要拆分成多组

                5. 示例结构（仅供参考，实际内容应基于给定文本）：
                Q: 什么是人工智能？ 
                A: 人工智能（Artificial Intelligence，简称AI）是指使计算机或机器模拟人类智能行为的技术和方法，包括学习、推理、规划、自然语言处理等能力。
                Q: 请解释一下人工智能的含义。 
                A: 人工智能（Artificial Intelligence，简称AI）是指使计算机或机器模拟人类智能行为的技术和方法，包括学习、推理、规划、自然语言处理等能力。
                Q: 人工智能的定义是什么？ 
                A: 人工智能（Artificial Intelligence，简称AI）是指使计算机或机器模拟人类智能行为的技术和方法，包括学习、推理、规划、自然语言处理等能力。
                
            给定文本：
            {chunk.page_content}

            请基于这个文本生成问答对。
            """
        response = get_completion(prompt)
        logger.debug(f"****response: {response}")
        if response:
            try:
                # 组装qa
                if response:
                    qa_map = packet_qa(qa_map, response)
                # parts = response.split("A:", 1)
                # if len(parts) == 2:
                #     question = parts[0].replace("Q:", "").strip()
                #     answer = parts[1].strip()
                #     qa_pairs.append({"question": question, "answer": answer})
                #     logger.debug(f"Generated QA pair for chunk question: {question}, answer: {answer}")
                #     logger.debug(f"*" * 50)
                else:
                    logger.warning(f"无法解析响应: {response}")
            except Exception as e:
                logger.warning(f"处理响应时出错: {traceback.format_exc()}")
        else:
            logger.debug(f"Failed to get completion for chunk {chunk}")
        
        progress = (i + 1) / len(text_chunks)
    
    return qa_map

def api_request(method, url, **kwargs):
    """通用API请求处理函数"""
    try:
        response = requests.request(method, url, headers=headers, **kwargs)
        response.raise_for_status()
        return response.json().get('data')
    except requests.RequestException as e:
        logger.debug(f"API请求失败: {e}")
        return None

def create_collection(name, embedding_model_id, capacity):
    """创建新集合"""
    data = {
        "name": name,
        "embedding_model_id": embedding_model_id,
        "capacity": capacity
    }
    return api_request("POST", f"{base_url}collections", json=data)

def create_chunk(collection_id, content):
    """创建chunk"""
    data = {
        "collection_id": collection_id,
        "content": content
    }
    endpoint = f"{base_url}collections/{collection_id}/chunks"  # 确保使用正确的端点
    try:
        response = requests.post(endpoint, headers=headers, json=data)
        response.raise_for_status()
        return response.json()['data']
    except requests.RequestException as e:
        logger.debug(f"创建chunk失败: {e}")
        return None
    
def load_single_document(file_path: str) -> List[Document]:
    """加载单个文档"""
    ext = "." + file_path.rsplit(".", 1)[-1]
    if ext in LOADER_MAPPING:
        loader_class, loader_args = LOADER_MAPPING[ext]
        loader = loader_class(file_path, **loader_args)
        return loader.load()
    raise ValueError(f"Unsupported file extension '{ext}'")

def process_document(documents, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:
    # 加载文档
    import spacy
    if not documents:
        logger.debug("文件处理失败，请检查文件格式是否正确。")
        return []

    # 初始化 SpacyTextSplitter
    nlp = spacy.load("zh_core_web_sm")
    text_splitter = SpacyTextSplitter(pipeline=nlp, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    # 分割文档
    text_chunks = []
    for document in documents:
        chunks = text_splitter.split_text(document.page_content)
        text_chunks.extend(chunks)
    print(text_chunks[0])
    return text_chunks


def process_file(uploaded_file, chunk_size=2000, chunk_overlap=500):
    """处理上传的文件并生成文本块"""
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name
    try:
        documents = load_single_document(tmp_file_path)
        if not documents:
            logger.debug("文件处理失败，请检查文件格式是否正确。")
            return []
        if config.use_recursive_splitter:
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, separators=["\n\n", "\n", " ", ""])
            text_chunks = text_splitter.split_documents(documents)
        else:
            text_chunks = process_document(documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        return text_chunks
    except Exception as e:
        logger.debug(f"处理文件时发生错误: {e}")
        return []
    finally:
        os.unlink(tmp_file_path)
    

def normalize_documents(documents):
    """"""
    # 替换掉连续的空格
    for document in documents:
        document.page_content = re.sub(r'\s+', ' ', document.page_content)
        logger.warning(f"normalized document: {document}")
    return documents

def process_files(uploaded_files:list, chunk_size=1000, chunk_overlap=200):
    """处理上传的多个文件并生成文本块"""
    all_text_chunks = []
    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file)[1]) as tmp_file:
            with open(uploaded_file, "rb") as f:
                file_content = f.read()
            tmp_file.write(file_content)
            tmp_file_path = tmp_file.name
        try:
            documents = load_single_document(tmp_file_path)
            logger.warning(f"documents: {len(documents[0].page_content)}")
            # 调用清洗函数，替换掉连续的空格、换行符和制表符
            # documents = normalize_documents(documents)
            if not documents:
                logger.debug(f"文件 {uploaded_file} 处理失败，请检查文件格式是否正确。")
                continue
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, separators=["\n\n"])
            text_chunks = text_splitter.split_documents(documents)
            print(f"*" * 50)
            # else:
            #     text_chunks = process_document(documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            all_text_chunks.extend(text_chunks)
        except Exception as e:
            logger.debug(f"处理文件 {uploaded_file} 时发生错误: {e}")
        finally:
            os.unlink(tmp_file_path)
    
    return all_text_chunks, documents[0]

def insert_qa_pairs_to_database(collection_id):
    """将问答对插入到数据库"""
    progress_bar = st.progress(0)
    status_text = st.empty()
    success_count = 0
    fail_count = 0
    for i, qa_pair in enumerate(st.session_state.qa_pairs):
        try:
            if "question" in qa_pair and "answer" in qa_pair and "chunk" in qa_pair:
                content = f"问题：{qa_pair['question']}\n答案：{qa_pair['answer']}\n原文：{qa_pair['chunk']}"
                # 如果content的字数超过4000，截取前4000字
                if len(content) > 4000:
                    content = content[:4000]  # 只保留前4000字
                if create_chunk(collection_id=collection_id, content=content):
                    success_count += 1
                else:
                    fail_count += 1
                    logger.warning(f"插入QA对 {i+1} 失败")
            else:
                fail_count += 1
                logger.warning(f"QA对 {i+1} 格式无效")
        except Exception as e:
            logger.debug(f"插入QA对 {i+1} 时发生错误: {str(e)}")
            fail_count += 1
        
        progress = (i + 1) / len(st.session_state.qa_pairs)
        progress_bar.progress(progress)
        status_text.text(f"进度: {progress:.2%} | 成功: {success_count} | 失败: {fail_count}")

    return success_count, fail_count

# Function to list chunks from a collection
def list_chunks(collection_id, limit=20, after=None):
    """List chunks from the specified collection."""
    url = f"{base_url}collections/{collection_id}/chunks"   
    params = {
        "limit": limit,
        "order": "desc"
    }
    if after:
        params["after"] = after

    response = api_request("GET", url, params=params)
    if response is not None:
        return response  # Assuming the response is a list of chunks
    else:
        logger.debug("列出 chunks 失败。")
        return []

# Function to get chunk details from a collection
def get_chunk_details(chunk_id, collection_id):
    """Get details of a specific chunk."""
    url = f"{base_url}collections/{collection_id}/chunks/{chunk_id}" 
    response = api_request("GET", url)
    if response is not None:
        return response  # Assuming the response contains chunk details
    else:
        logger.debug("获取 chunk 详细信息失败。")
        return None

# Function to fetch chunks from a collection
def fetch_all_chunks_from_collection(collection_id):
    """Fetch all chunks from the specified collection."""
    all_chunks = []
    after = None

    while True:
        chunk_list = list_chunks(collection_id, after=after)
        if not chunk_list:
            break
        # Get details for each chunk
        for chunk in chunk_list:
            chunk_id = chunk['chunk_id']
            chunk_details = get_chunk_details(chunk_id, collection_id)
            if chunk_details:
                all_chunks.append(chunk_details)
        # Check if we need to continue fetching
        if len(chunk_list) < 20:  # Assuming 20 is the limit
            break
        # Set the after parameter for the next request
        after = chunk_list[-1]['chunk_id']
    return all_chunks

# Function to download chunks as JSON
def download_chunks_as_json(chunks, collection_name):
    """Download chunks as a JSON file with clear formatting."""
    if chunks:
        json_data = {"chunks": []}
        for chunk in chunks:
            json_data["chunks"].append({
                "chunk_id": chunk.get("chunk_id"),
                "record_id": chunk.get("record_id"),
                "collection_id": chunk.get("collection_id"),
                "content": chunk.get("content"),
                "num_tokens": chunk.get("num_tokens"),
                "metadata": chunk.get("metadata", {}),
                "updated_timestamp": chunk.get("updated_timestamp"),
                "created_timestamp": chunk.get("created_timestamp"),
            })
        
        # Pretty print the JSON data for better readability
        json_str = json.dumps(json_data, ensure_ascii=False, indent=4)
        
        # Create a download button with the collection name
        st.download_button(
            label="下载集合内容为 JSON 文件",
            data=json_str,
            file_name=f"{collection_name}.json",
            mime="application/json"
        )

# Function to upload JSON chunks
def upload_json_chunks(uploaded_json_file, collection_id):
    """Upload chunks from a JSON file to the specified collection."""
    try:
        data = json.load(uploaded_json_file)
        
        if 'chunks' not in data:
            logger.debug("JSON 文件中缺少 'chunks' 键。")
            return
        
        chunks = data['chunks']
        total_records = len(chunks)
        records_per_collection = 1000
        num_collections = math.ceil(total_records / records_per_collection)

        st.write(f"总记录数: {total_records}")
        st.write(f"每个集合的记录数: {records_per_collection}")
        st.write(f"需要创建的集合数: {num_collections}")

        for i in range(num_collections):
            st.write(f"\n导入集合 {i+1}/{num_collections}...")
            start_index = i * records_per_collection
            end_index = min((i + 1) * records_per_collection, total_records)
            
            progress_bar = st.progress(0)
            for j, chunk in enumerate(chunks[start_index:end_index]):
                # Ensure the chunk has the required structure
                if 'content' in chunk:
                    content = chunk['content']
                    try:
                        create_chunk(
                            collection_id=collection_id,
                            content=content
                        )
                    except Exception as e:
                        logger.debug(f"创建 chunk 时出错: {str(e)}")
                        break
                else:
                    logger.warning(f"第 {start_index + j + 1} 条记录缺少 'content' 键。")
                    continue

                progress = (j + 1) / (end_index - start_index)
                progress_bar.progress(progress)

        st.success("所有数据导入完成。")
    except Exception as e:
        logger.debug(f"上传 JSON 文件时发生错误: {str(e)}")

def save_qa2csv(qa_pairs, save_path:str='qa_pairs.csv'):
    """
    
    """
    df = pd.DataFrame.from_dict(qa_pairs)
    df.to_csv(save_path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_url", type=str, default="https://api.deepseek.com/v1", help="API base URL")
    parser.add_argument("--operation", type=str, default="上传文件", help="选择操作")
    parser.add_argument("--option", type=str, default="", help="选择操作选项")
    parser.add_argument("--embedding", type=str, default="bge", help="Embedding model ID")
    parser.add_argument("--uploaded_files", type=str, nargs="+", help="上传的文件列表")
    parser.add_argument("--chunk_size", type=int, default=1000, help="Chunk size")
    parser.add_argument("--chunk_overlap", type=int, default=100, help="Chunk overlap")
    args = parser.parse_args()
    if args.operation == "上传文件":
        logger.debug("文件上传与QA对生成")
        # uploaded_file = st.file_uploader("上传非结构化文件", type=["txt", "pdf", "docx"])
        if args.uploaded_files:
            logger.debug("文件上传成功！, 开始处理文件...")
            
            text_chunks, documents = process_files(args.uploaded_files, chunk_overlap=args.chunk_overlap, chunk_size=args.chunk_size)
            if not text_chunks:
                logger.debug("文件处理失败，请检查文件格式是否正确。")
                return
            # print(text_chunks[0])
            logger.debug(f"文件已分割成 {len(text_chunks)} 个文本段")

            logger.debug("正在生成QA对...")
            # TODO: 输入全部文档内容
            qa_pairs = generate_qa_pairs_with_progress(text_chunks)

            # 临时保存QA对 到 csv 中
            save_qa2csv(qa_pairs)

            logger.debug(f"已生成 {len(qa_pairs['Q'])} 个QA对")
        else:
            logger.warning("请上传文件。")
    elif args.operation == "管理知识库":
        if args.option == "插入现有Collection":
            if st.session_state.collections:
                collection_names = [c['name'] for c in st.session_state.collections]
                selected_collection = st.selectbox("选择Collection", collection_names)
                selected_id = next(c['collection_id'] for c in st.session_state.collections if c['name'] == selected_collection)

                if st.button("插入QA对到选定的Collection"):
                    if hasattr(st.session_state, 'qa_pairs') and st.session_state.qa_pairs:
                        with st.spinner("正在插入QA对..."):
                            success_count, fail_count = insert_qa_pairs_to_database(selected_id)
                            st.success(f"数据插入完成！总计: {len(st.session_state.qa_pairs)} | 成功: {success_count} | 失败: {fail_count}")
                    else:
                        logger.warning("没有可用的QA对。请先上传文件并生成QA对。")
            else:
                logger.warning("没有可用的 Collections，请创建新的 Collection。")

        elif args.option == "创建新Collection":
            new_collection_name = st.text_input("输入新Collection名称")
            capacity = st.number_input("设置Collection容量", min_value=1, max_value=1000, value=1000)
            if st.button("创建新Collection"):
                with st.spinner("正在创建新Collection..."):
                    new_collection = create_collection(
                        name=new_collection_name,
                        embedding_model_id=args.embedding,  # 这里可以替换为实际的模型ID
                        capacity=capacity
                    )
                    if new_collection:
                        logger.debug(f"新Collection创建成功，ID: {new_collection['collection_id']}")
                        # 立即更新 collections 列表
                        collections = api_request("GET", f"{base_url}collections")
                    else:
                        logger.debug("创建新Collection失败")

        elif args.option == "下载Collection":
            if collections:
                collection_names = [c['name'] for c in collections]
                selected_collection = st.selectbox("选择Collection", collection_names)
                selected_id = next(c['collection_id'] for c in st.session_state.collections if c['name'] == selected_collection)

                if st.button("下载选定Collection的内容"):
                    with st.spinner("正在获取集合内容..."):
                        chunks = fetch_all_chunks_from_collection(selected_id)  # Pass the API key
                        if chunks:
                            download_chunks_as_json(chunks, selected_collection)  # Pass the collection name
                            st.success(f"成功获取 {len(chunks)} 个 chunk。")
                        else:
                            logger.debug("未能获取集合内容。")
            else:
                logger.warning("没有可用的 Collections，请创建新的 Collection。")

        elif args.option == "上传JSON文件":
            uploaded_json_file = st.file_uploader("选择一个 JSON 文件", type=["json"])
            
            if st.session_state.collections:
                collection_names = [c['name'] for c in st.session_state.collections]
                selected_collection = st.selectbox("选择Collection", collection_names)
                selected_id = next(c['collection_id'] for c in st.session_state.collections if c['name'] == selected_collection)

                if uploaded_json_file is not None:
                    if st.button("上传并插入到选定的Collection"):
                        with st.spinner("正在上传 JSON 文件并插入数据..."):
                            upload_json_chunks(uploaded_json_file, selected_id)
            else:
                logger.warning("没有可用的 Collections，请创建新的 Collection。")

if __name__ == "__main__":
    main()