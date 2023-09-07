import cv2
import os
import time
import json
import numpy as np
import datetime
import argparse
from test_cases import test_cases
from outputs import VideoResult
from frame_processor import vlm_processor,visualize_results
from dino_processor import dino_processor
import torch
from typing import Union
import matplotlib.pyplot as plt

def process_video_owl(
        video_path:str,
        text_queries=None,
        image_queries=None,
        interval=6,
        result_dir=None,
        max_frame=None,
        result_video=None,
        ):
    if text_queries is None and image_queries is None:
        print('Suply either image query or lang query')
        return
    if text_queries is not None and image_queries is not None:
        print('Suply either image query or lang query')
        return
    if text_queries is not None:
        query_type='lang'
    else:
        query_type='image'
    print(f"Searching with {query_type} Queries: {text_queries if text_queries else image_queries} in video {video_path}")
    device='cuda' if torch.cuda.is_available() else 'cpu'
    all_frame_results=[]
    vlm_processor.model.to(device)
    print("Using " + device)

    if query_type=='image':
        # convert BGR to RGB
        image_queries_cv2=[cv2.imread(name) for name in image_queries]
        image_queries_cv2=[cv2.cvtColor(i,cv2.COLOR_BGR2RGB) for i in image_queries_cv2]
    
    video = cv2.VideoCapture(video_path)

    if result_video is not None:
        #codec must be avc1 (h.264) to allowing playing in <video> element of html
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer=cv2.VideoWriter(result_video,fourcc,round(30.0/interval),(int(video.get(3)),int(video.get(4))))
    
    if result_dir is not None:
        os.makedirs(f'./results/{result_dir}',exist_ok=True)
    
    frame_count=0
    print(f"Number of frames: {int(video.get(cv2.CAP_PROP_FRAME_COUNT))} Processing interval: {interval}")

    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break
        #Very important: OpenCV read an image as BGR yet pytorch models assume an image is RGB
        frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        if max_frame is not None:
            if max_frame<=frame_count:
                break
        
        if frame_count%interval==0:
            start=time.perf_counter()
            if query_type=='lang':
                result=vlm_processor.process_image(frame,text_queries,device)
            else:
                result=vlm_processor.image_query(frame,image_queries_cv2,device)
            result['frame']=frame_count
            all_frame_results.append(result)
            end=time.perf_counter()
            print(f"Results of frame {frame_count}: {result['scores']} Time:{end-start}s")
            if query_type=='lang':
                class_string=", ".join([f"{index+1}->{c}" for index,c in enumerate(text_queries)])
                format_string=f"Classes: [{class_string}]"
                visualized_image=visualize_results(frame,result,text_queries,format_string)
            else:
                format_string=f"Image: {', '.join(image_queries)}"
                visualized_image=visualize_results(frame,result,image_queries,format_string)
            if result_dir is not None:
                result_path=f'./results/{result_dir}/frame_{frame_count}.jpg'
                cv2.imwrite(result_path,visualized_image)
            if result_video is not None:
                video_writer.write(visualized_image)
        else:
            all_frame_results.append({'frame':frame_count})
        frame_count=frame_count+1
    if result_video is not None:
        video_writer.release()
    video.release()
    return all_frame_results

def process_video_gdino(
        video_path:str,
        text_queries=None,
        image_queries=None,
        interval=6,
        result_dir=None,
        max_frame=None,
        result_video=None,
):
    if text_queries is None and image_queries is None:
        print('Suply either image query or lang query')
        return
    if text_queries is not None and image_queries is not None:
        print('Suply either image query or lang query')
        return
    if text_queries is not None:
        query_type='lang'
    else:
        query_type='image'
    print(f"Searching with {query_type} Queries: {text_queries if text_queries else image_queries} in video {video_path}")
    device='cuda' if torch.cuda.is_available() else 'cpu'
    all_frame_results=[]
    print("Using " + device)
    video = cv2.VideoCapture(video_path)
    if result_video is not None:
        #codec must be avc1 (h.264) to allowing playing in <video> element of html
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer=cv2.VideoWriter(result_video,fourcc,round(30.0/interval),(int(video.get(3)),int(video.get(4))))
    
    if result_dir is not None:
        os.makedirs(f'./results/{result_dir}',exist_ok=True)
    
    frame_count=0
    print(f"Number of frames: {int(video.get(cv2.CAP_PROP_FRAME_COUNT))} Processing interval: {interval}")

    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break
        #Very important: OpenCV read an image as BGR yet pytorch models assume an image is RGB
        frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        if max_frame is not None:
            if max_frame<=frame_count:
                break
        
        if frame_count%interval==0:
            start=time.perf_counter()
            if query_type=='lang':
                result,visualized_image=dino_processor.process_image(frame,text_queries,visualize=result_video is not None)
                # plt.imshow(visualized_image)
                # exit()
            else:
                result,visualized_image=dino_processor.image_query(frame,image_queries_cv2)
            result['frame']=frame_count
            all_frame_results.append(result)
            end=time.perf_counter()
            print(f"Results of frame {frame_count}: {result['scores']} Time:{end-start}s")
            if result_dir is not None:
                result_path=f'./results/{result_dir}/frame_{frame_count}.jpg'
                cv2.imwrite(result_path,visualized_image)
            if result_video is not None:
                video_writer.write(visualized_image)
        else:
            all_frame_results.append({'frame':frame_count})
        frame_count=frame_count+1
    if result_video is not None:
        video_writer.release()
    video.release()
    return all_frame_results

processor_mapping={
    'owl-vit':process_video_owl,
    'gdino':process_video_gdino,
}

#TODO: create QueryExecutor class
def run_video(
            video_name:str,
            queries:list[str],
            run_type:str,
            interval:int=3,
            max_frame:Union[int,None]=None,
            visualize_all:bool=False,
            top_k:Union[int,None]=None,
            chunk_size:Union[int,None]=None,
            model_name='owl-vit',
            ):
    print(f"Using model {model_name}")
    os.makedirs('results',exist_ok=True)
    video_raw_name=video_name.split('/')[-1]
    str_max_frame=''
    if max_frame is not None:
        str_max_frame=f'_frame_{max_frame}'
    #append time stamp into name to avoid duplication
    current_time = datetime.datetime.now()
    formatted_time = current_time.strftime("%Y%m%d%H%M")
    if visualize_all:
        result_video_name=f'./results/{video_raw_name}{str_max_frame}_{formatted_time}_{run_type}.mp4'
    else:
        result_video_name=None
    result_json_name=f'./results/{video_raw_name}{str_max_frame}_{formatted_time}_{run_type}.json'
    print(f'Result Video Path: {result_video_name}')
    if run_type not in ['image','lang']:
        print("Invalid run_type: must be image or lang ")
        return
    process_video_method=processor_mapping[model_name]
    if run_type=='image':
        result=process_video_method(video_name,image_queries=queries,interval=interval,result_dir=None,max_frame=max_frame,result_video=result_video_name)
    elif run_type=='lang':
        result=process_video_method(video_name,text_queries=queries,interval=interval,result_dir=None,max_frame=max_frame,result_video=result_video_name)
    result={
            'query':queries,
            'type':run_type,
            'result':result,
        }
    with open(result_json_name,'w') as f:
        print(f"Result Json: {result_json_name}")
        json.dump(result,f)
    if top_k is not None:
        video_result=VideoResult()
        video_result.from_data_dict(result)
        sorted_chunks_ma=video_result.sort_logits_chunks_ma(chunk_size)
        result_dirs=video_result.dump_top_k_chunks(video_name,sorted_chunks_ma,top_k)
        return result_dirs
    else:
        return []



    
#If you are using CUHK CSE slurm cluster
#export SLURM_CONF=/opt1/slurm/gpu-slurm.conf 
#srun --gres=gpu:1 -w gpu39 --pty /bin/bash
if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--video_name',type=str)
    parser.add_argument('--query_index',type=int)
    parser.add_argument('--max_frame',type=int,default=None)
    parser.add_argument('--interval',type=int,default=10,help="the number of frame between every model execution")
    parser.add_argument('--visualize_all', action='store_true', default=False,help='visualize all bounding boxes of the video')
    parser.add_argument('--top_k',type=int,default=None,help="top k chunks to output, if None, no chunk will be output")
    parser.add_argument('--chunk_size',type=int,default=None,help="Number of frames in a chunk") # 2 seconds
    parser.add_argument('--model_name',type=str,default=None,choices=['owl-vit','gdino']) 
    args=parser.parse_args()
    video_name=args.video_name
    query=test_cases[args.query_index]['object']
    query_type=test_cases[args.query_index]['type']
    results_dirs=run_video(video_name,query,query_type,interval=args.interval,visualize_all=args.visualize_all,top_k=args.top_k,chunk_size=args.chunk_size,model_name=args.model_name)
    print(f"Results saved to {results_dirs}")



    
