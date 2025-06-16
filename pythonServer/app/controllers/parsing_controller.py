from flask import Blueprint, request, jsonify
from app.services.code_parser import parse_code

parsing_bp = Blueprint('parsing', __name__)

@parsing_bp.route('/parse', methods=['POST'])
def handle_parse_request():
    data = request.get_json()
    if not data or 'code' not in data:
        return jsonify({
            'success': False,
            'error': 'Missing code in request'
        }), 400

    try:
        result, status_code = parse_code(data['code'])

        # Expect interpreter to return {status: ..., results: [...]}
        # Prepare output for text, table, structure
        text_out = ""
        table_out = []
        structure_out = None

        if isinstance(result, dict) and "results" in result:
            # results is a list of dicts, each with "status", "message", possibly others
            text_out = "\n\n".join(str(item.get("message", item)) for item in result["results"])
            table_out = result["results"]
            structure_out = result
        else:
            # fallback for error or other shape
            text_out = str(result)
            structure_out = result

        return jsonify({
            'success': True,
            'result': {
                'text': text_out,
                'table': table_out,
                'structure': structure_out
            }
        }), status_code
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500
