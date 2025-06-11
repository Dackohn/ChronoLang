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
        result = parse_code(data['code'])
        
        return jsonify({
            'success': True,
            'result': result[0]['result'], 
            'ast': {} 
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500