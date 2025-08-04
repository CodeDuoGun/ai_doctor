from app.service.chat import chat_service, judge_intent
from app.service.conversation import create_conversation, get_conversation, delete_conversation, update_conversation
from app.service.es_service import udpate_qa_knowledge
import time
from flask import Blueprint, Response, stream_with_context, request
from app.utils.log import logger
from app.utils.response import general_response
from app.schema.chat import ChatRequest, CreateChatItem, IntentItem, UpdateQAItem
import traceback
from app.config import config


bp = Blueprint('chat', __name__)
# from sentence_transformers import SentenceTransformer
# bge_embedding_model = SentenceTransformer("BAAI/bge-large-zh-v1.5")
bge_embedding_model = None
if config.USE_RANK:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    tokenizer = AutoTokenizer.from_pretrained("app/model/bge-reranker-large", local_files_only=True)
    reranker = AutoModelForSequenceClassification.from_pretrained("app/model/bge-reranker-large", local_files_only=True)
    reranker.eval()
    print(f"load model success")

    # from FlagEmbedding import FlagReranker
    # reranker = FlagReranker('BAAI/bge-reranker-large', use_fp16=True, devices="cpu")  # Setting use_fp16 to True speeds up computation with a slight performance degradation
else: 
    reranker = None
    tokenizer = None


@bp.route('/aichat', methods=['POST'])
def chat():
    """
    聊天接口
    """
    # TODO: @txueduo 需要确认是否需要对请求体进行校验，如果没有这个会话,应该返回错误
    try:
        backend_id = request.headers.get("trace_id", "")
        chat_params = ChatRequest(**request.json)
        logger.info(f"trace_id: {backend_id} receive params : {request.json}")
        #TODO: 流式返回 refer: https://www.coze.cn/open/docs/developer_guides/chat_v3#9a377f07
        return Response(
            stream_with_context(chat_service(chat_params, backend_id, bge_embedding_model, reranker, tokenizer)),content_type='text/event-stream;charset=utf-8'
            )
    except Exception as e:
        logger.error(f"chat error for {e}")
        return general_response(code=500, msg=f"chat error for {e}")

@bp.route('/intent', methods=['POST'])
def get_intent():
    """0,"""
    try:
        params = IntentItem(**request.json)
        backend_id = request.headers.get("trace_id", "")
        trace_id = f"{backend_id}_{params.conversation_id}"
        data = judge_intent(params.query, trace_id)
        return general_response(data=data)
    except Exception as e:
        logger.error(f"意图识别失败 {traceback.format_exc()}")
        return general_response(code=500, msg=f"意图识别失败{e}")
    

# 创建会话，返回会话id
@bp.route('/conversation/create', methods=['POST'])
def create_convers():
    """"""
    try:
        params = CreateChatItem(**request.json)
        backend_id = request.headers.get("trace_id", "")
        user_id = params.uid
        # chat_id = int(time.time())
        chat_id = "uid"
        conversation_id = create_conversation(chat_id, user_id, params.device_id)
        logger.info(f"trace_id: {backend_id}_{conversation_id}, success created")
        return general_response(data={"id": conversation_id})
    except Exception as e:
        logger.error(f"{backend_id}_{conversation_id} 创建会话失败 {traceback.format_exc()}")
        return general_response(code=500, msg=f"创建会话失败:{e}")

@bp.route('/conversation/search', methods=['POST'])
def search_convers():
    """"""
    try:
        request_body = request.json
        history= get_conversation(request_body['conversation_id']) 
        return general_response(data={"id": request_body["conversation_id"],"history": history})
    except Exception as e:
        return general_response(code=500, msg=f"获取会话{request_body['conversation_id']}失败:{e}")

@bp.route('/conversation/delete', methods=['POST'])
def delete_convers():
    """从数据库中彻底删除会话"""
    try:
        request_body = request.json
        delete_conversation(request_body['conversation_id']) 
        return general_response(data={"id": request_body['conversation_id']})
    except Exception as e:
        return general_response(code=500, msg=f"删除会话{request_body['conversation_id']}, 失败:{e}")

@bp.route('/conversation/<string:conversation_id>/clear', methods=['POST'])
def clear_context(conversation_id):
    """清除会话上下文"""
    try:
        backend_id = request.headers.get("trace_id", "")
        trace_id = f"{backend_id}_{conversation_id}"
        update_conversation(conversation_id, new_history=[], trace_id=trace_id) 
        return general_response(data={"id": conversation_id})
    except Exception as e:
        return general_response(code=500, msg=f"清除会话上下文失败:{e}")

# TODO: 针对es数据库的一些接口
@bp.route('/es/update_qa', methods=['POST'])
async def update_es():
    """
    更新 qa
    """
    try:
        backend_id = request.headers.get("trace_id", "")
        params = UpdateQAItem(**request.json)
        udpate_qa_knowledge(params.question, params.answer, params.new_question, params.new_answer)

        return general_response()
    except Exception as e:
        return general_response(code=500, msg=f"清除会话上下文失败:{e}")