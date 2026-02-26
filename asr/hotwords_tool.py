from dashscope.audio.asr import VocabularyService, vocabulary
from utils.log import logger
import json
from typing import List
import dashscope
from dotenv import load_dotenv
import os

load_dotenv()

class HotwordsManagement():
    def __init__(self) -> None:
        
        self.service = VocabularyService()
        dashscope.api_key = os.environ.get('DASHSCOPE_API_KEY')

        # 以下为北京地域url，若使用新加坡地域的模型，需将url替换为：https://dashscope-intl.aliyuncs.com/api/v1
        dashscope.base_http_api_url = 'https://dashscope.aliyuncs.com/api/v1'

    

    def _create(self, prefix:str, target_model:str, my_vocabulary:list):
        """
        param: target_model 热词列表对应的语音识别模型，必须与后续调用语音识别接口时使用的语音识别模型一致
        param: prefix 热词列表自定义前缀，仅允许数字和小写字母，小于十个字符。
        param: vocabulary 热词列表JSON
        return: 热词列表ID
        """
        vocabulary_id = self.service.create_vocabulary(
            prefix=prefix,
            target_model=target_model,
            vocabulary=my_vocabulary)
        logger.info(f"create vocab {vocabulary_id} success")


    def _query(self,vocab_id:str):
        """"""
        vocabulary = self.service.query_vocabulary(vocab_id)

    def _list(self, prefix=None, page_index: int = 0, page_size: int = 10) -> List[dict]:
        '''
        查询已创建的所有热词列表
        param: prefix 自定义前缀，如果设定则只返回指定前缀的热词列表标识符列表。
        param: page_index 查询的页索引
        param: page_size 查询页大小
        return: 热词列表标识符列表
        '''
        vocabularies = self.service.list_vocabularies()
        logger.info(f"热词列表：{json.dumps(vocabularies)}")
        return vocabularies

    def _update(self,my_vocabulary:list, vocab_id:str):
        """
        my_vocabulary = [
            {"text": "赛德克巴莱", "weight": 4, "lang": "zh"}
        ]
        """
        self.service.update_vocabulary(vocab_id, my_vocabulary)

    def _delete(self, id):
        self.service.delete_vocabulary(id)
        logger.info(f"success del {id}")
    

    

if __name__ == "__main__":
    manager = HotwordsManagement()
    target_model = "fun-asr-realtime"
    import glob
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("operator")
    args = parser.parse_args()
    count = 0
    # vocab-shylasr0-04d02cc8f3714a5d805f0a45955776c0  vocab-shylasr0-d93cdf71371e41fd94285a4ea7a4f7f9
    if args.operator == "create":
        for json_file in sorted(glob.glob("asr/*.json")):
            prefix = f"shylasr{count}"
            print(json_file, prefix)
            # break
            with open(f"{json_file}", "r", encoding="utf-8") as f:
                vocabulary = json.load(f)
                
                manager._create(prefix=prefix, target_model=target_model, my_vocabulary=vocabulary)
    elif args.operator == "del":
        vbs = manager._list()
        for vb in vbs:
            manager._delete(vb["vocabulary_id"])
            print(f"delete {vb['vocabulary_id']} success")
    elif args.operator == "list":
        vbs = manager._list() 
    elif args.operator == "update":
        json_file = "asr/hotwords_09.json"
        with open(f"{json_file}", "r", encoding="utf-8") as f:
            vocabulary = json.load(f)
            manager._update(vocabulary, "vocab-shylasr0-df85d72b342747309b91a7c84c53fdf7")
        
