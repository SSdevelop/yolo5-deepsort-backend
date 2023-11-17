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
from werkzeug.utils import secure_filename
import torch

checkpoint = "google/owlvit-base-patch32"
UPLOAD_FOLDER = 'results/temp'
app = Flask(__name__)
CORS(app)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

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
        # if not filepath.endswith('.mp4'):
        #     filepath = f'{filepath}.mp4'
        pathobj = Path(filepath[2:])
        if not pathobj.exists():
            return jsonify({'message': 'File not found'}), 404
        response = send_file(os.path.join(filepath[2:]), mimetype='video/webm')
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
            file = request.files.getlist('image_query')
            result_dirs = process_image_query(body, file)
            return jsonify({'message': result_dirs}), 200
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

def process_image_query(body, file):
    os.makedirs('results', exist_ok=True)
    os.makedirs('results/temp', exist_ok=True)
    interval = 5 
    video_names_temp = body['video_names']
    if not isinstance(video_names_temp, list):
        video_names = [video_names_temp]
    else:
        video_names = video_names_temp
    images = file
    print(images)
    image_names = [each_file.filename for each_file in images]  
    filenames = [secure_filename(imagename) for imagename in image_names]
    # (image.save(os.path.join(UPLOAD_FOLDER, (filename for filename in filenames))) for image in images)
    # (image.save(os.path.join(UPLOAD_FOLDER)) for image in images)
    for image in images:
        image.save('results/temp/'+image.filename)
    images=[cv2.imread(f'{UPLOAD_FOLDER}/{name}') for name in image_names]
    print(filenames)
    print(f'image name {image_names}, video name {video_names} filename {filenames}')
    device='cuda' if torch.cuda.is_available() else 'cpu'
    frame_processor = FrameProcessor()
    result_dirs = []
    for video_name in video_names:
        # print("name", video_name)
        all_frame_results=[]
        video = cv2.VideoCapture(f'utils/videos/{video_name}')
        # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        # os.makedirs(f'results/{video_name}', exist_ok=True)
        frame_count=0
        print(f"Number of frames: {int(video.get(cv2.CAP_PROP_FRAME_COUNT))} Processing interval: {interval}")
        while video.isOpened():
            ret, frame = video.read()
            if not ret:
                break
            frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            if frame_count%interval==0:
                result=frame_processor.image_query(frame,images[0])
                result['frame']=frame_count
                print(result)
                all_frame_results.append(result)
            else:
                all_frame_results.append({'frame':frame_count})
            frame_count+=1
        video.release()
        results = {
            'query':image_names,
            'type':'image',
            'result':all_frame_results,
        }
        video_result=VideoResult()
        video_result.from_data_dict(results)
        sorted_chunks_ma=video_result.sort_logits_chunks_ma(90)
        result_dir=video_result.dump_top_k_chunks(video_name,sorted_chunks_ma,5)
        result_dirs.append(result_dir[0])
    return result_dirs

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
        print(sorted_chunks_ma)
        # for k in sorted_chunks_ma:
        #     sorted_chunks_ma[k]=sorted_chunks_ma[k]
        result_dir=video_result.dump_top_k_chunks(video_name,sorted_chunks_ma,3)
        result_dirs.append(result_dir[0])
    return result_dirs
if __name__ == '__main__':
    #start the flask app
    app.run(debug=True)