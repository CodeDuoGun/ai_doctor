from langchain.text_splitter import RecursiveCharacterTextSplitter
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
from langchain.docstore.document import Document
from typing import List
import os
from utils.log import logger


LOADER_MAPPING = {
    "csv": (CSVLoader, {}),
    "doc": (UnstructuredWordDocumentLoader, {}),
    "docx": (UnstructuredWordDocumentLoader, {}),
    "enex": (EverNoteLoader, {}),
    "eml": (UnstructuredEmailLoader, {}),
    "epub": (UnstructuredEPubLoader, {}),
    "html": (UnstructuredHTMLLoader, {}),
    "md": (UnstructuredMarkdownLoader, {}),
    "odt": (UnstructuredODTLoader, {}),
    "pdf": (PyMuPDFLoader, {}),
    "ppt": (UnstructuredPowerPointLoader, {}),
    "pptx": (UnstructuredPowerPointLoader, {}),
    "txt": (TextLoader, {"encoding": "utf8"}),
}

# TODO: @txueduo 兼容其他文件格式
def load_single_document(file_path: str) -> List[Document]:
    """加载单个文档"""
    ext = os.path.basename(file_path).split(".")[-1].lower()
    if ext in LOADER_MAPPING:
        loader_class, loader_args = LOADER_MAPPING[ext]
        loader = loader_class(file_path, **loader_args)
        return loader.load()
    raise ValueError(f"Unsupported file extension '{ext}'")

def load_and_split_docx(file_path: str, chunk_size: int=1000, chunk_overlap:int=0) -> List[Document]:
    """
    加载 docx 文件，并利用递归切分器生成文档块
    """
    docs = load_single_document(file_path)
    # TODO: @txueduo 使用其他的切割方法 
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, separators=["\n\n"])
    documents = text_splitter.split_documents(docs)
    logger.info(f"success split file {file_path} to {len(documents)} chunks.")
    return documents
