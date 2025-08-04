from flask import Blueprint
import traceback
bp = Blueprint('status', __name__)
from app.utils.response import general_response
from app.utils.log import logger

# TODO: 针对es数据库的一些接口
@bp.route('/status', methods=['GET'])
def get_status():
    """
    test status
    """
    try:
        res = general_response()
    except Exception:
        logger.error(f"app exit for {traceback.format_exc()}") 
    return res