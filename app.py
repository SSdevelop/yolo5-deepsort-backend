import json
from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
from model.lf import run_video_to_video
from model.outputs import VideoResult
import logging
import os

app = Flask(__name__)
CORS(app)

logging_level = logging.DEBUG
logging.basicConfig(level=logging_level,format='[%(lineno)d]:[%(asctime)s]:%(message)s')

@app.route('/files/<filename>', methods=['GET'])
def get_file(filename):
    response = send_file(os.path.join('utils', 'videos', filename))
    return response

@app.route('/upload', methods=['POST'])
def inference():
    try:
        body = dict(request.get_json())
        if body is None:
            return jsonify({'message': 'No query type provided'}), 400
        video_names = body['video_names']
        if video_names is None or len(video_names) == 0:
            return jsonify({'message': 'No video names provided'}), 400
        query_type = body['query_type']
        if query_type == 'image':
            process_image_query(body)
        elif query_type == 'lang':
            process_lang_query(body)
        else:
            return jsonify({'message': 'Invalid query type'}), 400
    
    except Exception as e:
        logging.error(e)
        return jsonify({'message': 'Something went wrong', 'error': str(e)}), 500

if __name__ == '__main__':
    #start the flask app
    app.run(debug=True)