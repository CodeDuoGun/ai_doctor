import os
import json
from constants import LLMType

class ReportProcessor():
    """
    检查报告解读，报告的格式可能有多种
    pdf word jpeg png dicom(目前不支持）

    """
    def __init__(self):
        pass

    def read_report(self,report_path:str):
        report_type = os.path.basename(report_path).split(".")[-1]
        if report_type == "pdf":
            pass
        elif report_type in ("jpg", "png", "jpeg"):
            pass
        else:
            pass

    def read_by_vlm(self, vlm_type):
        if vlm_type == LLMType.Qwen:
            pass
        elif vlm_type == LLMType.DouBao:
            pass
        elif vlm_type == LLMType.GPT:
            pass
            
        

    def read_by_llm(self):
        pass
    
    def preprocess_data(self):
        pass

def convert_pdf2text(pdf_path:str, txt_path:str):
    """"""

    pass



    




        