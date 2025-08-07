from utils.log import logger
import traceback
from ai_emr_customer.app.rag.split import load_and_split_docx


def save2csv():

    pass


def insert2es(self):
    pass


def extract_pdf2text(pdf_file):
    documents = load_and_split_docx(pdf_file, chunk_size=1000, chunk_overlap=20)
    # 大模型抽取知识信息
    extract_res = []
    try:
        for doc in documents:
            pass
            
    except Exception:
        logger.error(f"extract doc err {traceback.format_exc()}")
        res = []


    pass

