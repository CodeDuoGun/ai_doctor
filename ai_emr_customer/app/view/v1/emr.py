from flask import Blueprint
import traceback
bp = Blueprint('status', __name__)
from app.utils.response import general_response
from app.utils.log import logger
from app.service.emr import gen_ai_emr
from app.schema.emr import EmrRequestSchema


@bp.route('/emr', methods=['POST'])
def get_ai_emr(item: EmrRequestSchema):
    """
    ai 预问诊，之后根据内容，生成ai病历
    """
    try:
        res = general_response()
    except Exception:
        logger.error(f"app exit for {traceback.format_exc()}") 
    return res

