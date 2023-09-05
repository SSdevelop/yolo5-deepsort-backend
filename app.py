import cv2
from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
from model.owl.lf import run_video_to_video
from model.owl.outputs import VideoResult
from model.frame_processor import FrameProcessor, visualize_results
import logging
import os

checkpoint = "google/owlvit-base-patch32"

app = Flask(__name__)
CORS(app)

logging_level = logging.DEBUG
logging.basicConfig(level=logging_level,format='[%(lineno)d]:[%(asctime)s]:%(message)s')

@app.route('/files/<filename>', methods=['GET'])
def get_file(filename):
    response = send_file(os.path.join('utils', 'videos', filename))
    return response

@app.route('/files/results/<filename>', methods=['GET'])
def get_result_file(filename):
    response = send_file(os.path.join('results', filename))
    return response

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
            process_lang_query(body)
        else:
            return jsonify({'message': 'Invalid query type'}), 400
    
    except Exception as e:
        logging.error(e)
        return jsonify({'message': 'Something went wrong', 'error': str(e)}), 500

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

if __name__ == '__main__':
    #start the flask app
    app.run(debug=True)