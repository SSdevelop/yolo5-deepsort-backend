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
        query_type = request.form.get('query_type')
        if query_type is None or len(query_type) == 0:
            return jsonify({'message': 'No query type provided'}), 400
        video_names = request.form.getlist('video_names')
        if video_names is None or len(video_names) == 0:
            return jsonify({'message': 'No video names provided'}), 400
        queries = None
        if query_type == 'image':
            image_query = request.files.getlist('image_query')
            if image_query is None or len(image_query) == 0:
                return jsonify({'message': 'No image query provided'}), 400
            logging.info('Images query received')
            images = []
            for i in range(len(image_query)):
                images.append(image_query[i].filename)
                image_query[i].save(os.path.join('utils', 'images', image_query[i].filename))
            logging.info('Images saved')
            logging.info(f'Images: {images}')
            for i in range(len(image_query)):
                images[i] = f'../utils/images/{images[i]}'
            logging.info(f'Images with relative path: {images}')
            queries = images
        elif query_type == 'lang':
            lang_query = request.form.get('lang_query')
            if lang_query is None or len(lang_query) == 0:
                return jsonify({'message': 'No lang query provided'}), 400
            logging.info('Language query received')
            queries = lang_query.split(',')
        else:
            return jsonify({'message': 'Invalid query type'}), 400
        
        logging.info('Starting the Model with the query')
        video_dir = []
        for video_name in video_names:
            json_dump_file = run_video_to_video(f'utils/videos/{video_name}', queries, run_type=query_type)
            json_dump_file = f'./model/{json_dump_file[2:]}'
            logging.info(f'Model finished for {video_name}')
            logging.info(f'JSON file: {json_dump_file}')
            video_results = VideoResult()
            with open(json_dump_file) as f:
                json_data=json.load(f)
                video_results.from_data_dict(json_data)
            sorted_results = video_results.sort_logits_chunks_ma(60)
            for k in sorted_results:
                sorted_results[k]=sorted_results[k][:5]
            logging.info(f'Sorted results: {sorted_results}')
            save_dir=video_results.dump_top_k_chunks(f'utils/videos/{video_name}',sorted_results,5)
            video_dir.append(save_dir)
            logging.info(f'Video dir for {video_name}: {save_dir}')
        return jsonify({'video_dir': video_dir}), 200
    except Exception as e:
        logging.error(e)
        return jsonify({'message': 'Something went wrong', 'error': str(e)}), 500

if __name__ == '__main__':
    #start the flask app
    app.run(debug=True)