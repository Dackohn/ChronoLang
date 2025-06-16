import subprocess
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

def parse_code(code: str):
    try:
        base_dir = Path(__file__).parent.parent.parent
        interpreter_path = base_dir / 'cl' / 'Interpreter' / 'test.py'
        
        if not interpreter_path.exists():
            error_msg = f"Interpreter not found at {interpreter_path}"
            logger.error(error_msg)
            return {'error': error_msg}, 500
        
        result = subprocess.run(
            ['python', str(interpreter_path), code],
            capture_output=True,
            text=True,
            timeout=30 
        )
        
        logger.debug(f"Interpreter stdout: {result.stdout}")
        logger.debug(f"Interpreter stderr: {result.stderr}")
        
        if result.returncode == 0:
            return {'result': result.stdout.strip()}, 200
        else:
            return {'error': result.stderr.strip() or 'Interpreter failed'}, 400
            
    except subprocess.TimeoutExpired:
        error_msg = "Interpreter timed out"
        logger.error(error_msg)
        return {'error': error_msg}, 408
        
    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return {'error': error_msg}, 500