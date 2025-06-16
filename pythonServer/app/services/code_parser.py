import subprocess
from pathlib import Path
import logging
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'cl', 'Interpreter'))


from Interpreter import execute_chronolang_json

logger = logging.getLogger(__name__)

def parse_code(code: str):
    try:

        import json
        json_input = json.dumps({"code": code})

        result = execute_chronolang_json(json_input)
        
        logger.debug(f"Interpreter result: {result}")

        if result.get('status') == 'error':
            error_msg = result.get('message', 'Unknown error')
            if 'traceback' in result:
                logger.error(f"Interpreter traceback: {result['traceback']}")
            return {'error': error_msg}, 400
        elif result.get('status') == 'success':
            return {'result': result.get('result', result.get('message', ''))}, 200
        else:

            return {'result': str(result)}, 200
            
    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return {'error': error_msg}, 500