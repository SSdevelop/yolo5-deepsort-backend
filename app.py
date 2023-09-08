import cv2
from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
# from model.owl.lf import run_video_to_video
from model.owl.outputs import VideoResult
from model.frame_processor import FrameProcessor, visualize_results
from model.dino_processor import dino_processor
import logging
import os
from pathlib import Path

checkpoint = "google/owlvit-base-patch32"

app = Flask(__name__)
CORS(app)

logging_level = logging.DEBUG
logging.basicConfig(level=logging_level,format='[%(lineno)d]:[%(asctime)s]:%(message)s')

@app.route('/files/<filename>', methods=['GET'])
def get_file(filename):
    try:
        if not filename.endswith('.mp4'):
            filename = f'{filename}.mp4'
        if not os.path.exists(os.path.join('utils', 'videos', filename)):
            return jsonify({'message': 'File not found'}), 404
        response = send_file(os.path.join('utils', 'videos', filename))
        return response
    except Exception as e:
        logging.error(e)
        return jsonify({'message': 'Something went wrong', 'error': str(e)}), 500

@app.route('/files/results', methods=['GET'])
def get_result_file():
    try:
        filepath = request.args.get('filepath')
        print(filepath[2:])
        if not filepath.endswith('.mp4'):
            filepath = f'{filepath}.mp4'
        pathobj = Path(filepath[2:])
        if not pathobj.exists():
            return jsonify({'message': 'File not found'}), 404
        response = send_file(os.path.join(filepath[2:]), mimetype='video/mp4')
        return response
    except Exception as e:
        logging.error(e)
        return jsonify({'message': 'Something went wrong', 'error': str(e)}), 500

@app.route('/upload', methods=['POST'])
def inference():
    try:
        body = request.form
        if body is None:
            return jsonify({'message': 'No query type provided'}), 400
        video_names = body['video_names']
        if video_names is None or len(video_names) == 0:
            return jsonify({'message': 'No video names provided'}), 400
        query_type = body['query_type']
        if query_type == 'image':
            # images = body['images']
            process_image_query(body)
            return jsonify({'message': 'Success'}), 200
        elif query_type == 'lang':
            result_dirs = process_lang_query(body)
            return jsonify({'message': result_dirs}), 200
        else:
            return jsonify({'message': 'Invalid query type'}), 400
    
    except Exception as e:
        logging.error(e.with_traceback())
        return jsonify({'message': 'Something went wrong', 'error': str(e)}), 500
    
@app.route('/hello', methods=['GET'])
def hello():
    return jsonify({'message': 'Hello'}), 200

def process_image_query(body):
    os.makedirs('results', exist_ok=True)
    interval = 5
    video_names = body['video_names']
    images = body['images']
    image_names = [os.path.basename(image) for image in images]
    images=[cv2.imread(name) for name in images]
    images=[cv2.cvtColor(i,cv2.COLOR_BGR2RGB) for i in images]
    frame_processor = FrameProcessor(checkpoint)
    for video_name in video_names:
        video = cv2.VideoCapture(os.path.join('utils', 'videos', video_name))
        # os.makedirs(os.path.join('results'), exist_ok=True)
        fps = video.get(cv2.CAP_PROP_FPS)
        video_writer = cv2.VideoWriter(os.path.join('results', f'{video_name}.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), fps, (int(video.get(3)),int(video.get(4))))
        frame_count = 0
        while video.isOpened():
            ret, frame = video.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if frame_count % interval == 0:
                frame_results = frame_processor.image_query(image=frame, image_query=images)
                frame_results['frame'] = frame_count
                visualized_frame = visualize_results(frame, frame_results, f"Image: {', '.join(image_names)}")
                video_writer.write(visualized_frame)
            else:
                video_writer.write(frame)
            frame_count += 1
        video_writer.release()
        video.release()

def process_lang_query(body):
    os.makedirs('results', exist_ok=True)
    lang_query = [body['lang_query']]
    video_names = body['video_names'].split(',')
    result_dirs = []
    for video_name in video_names:
        video = cv2.VideoCapture(os.path.join('utils', 'videos', video_name))
        # os.makedirs(os.path.join('results'), exist_ok=True)
        fps = video.get(cv2.CAP_PROP_FPS)
        frame_count = 0
        video_results = []
        while video.isOpened():
            ret, frame = video.read()
            if not ret:
                break

            frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            if frame_count % 6 == 0:
                result,visualized_image=dino_processor.process_image(frame,lang_query,visualize=True)
                result['frame']=frame_count
                video_results.append(result)
            else:
                video_results.append({'frame':frame_count})
            frame_count += 1
        video.release()
        results = {
            'query':lang_query,
            'type':'lang',
            'result':video_results,
        }
        print(results)
        video_result=VideoResult()
        video_result.from_data_dict(results)
        sorted_chunks_ma=video_result.sort_logits_chunks_ma(90)
        # for k in sorted_chunks_ma:
        #     sorted_chunks_ma[k]=sorted_chunks_ma[k]
        result_dir=video_result.dump_top_k_chunks(video_name,sorted_chunks_ma,3)
        result_dirs.append(result_dir[0])
    return result_dirs
if __name__ == '__main__':
    #start the flask app
    app.run(debug=True)