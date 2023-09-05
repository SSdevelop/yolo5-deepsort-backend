import matplotlib.pyplot as plt
import json
import numpy as np
from outputs import VideoResult
from utils.utils import file_name,split_suffix
from utils.fn import max_with_default
import numpy as np

def plot_frame_score(json_path):
    video_results=VideoResult()
    with open(json_path) as f:
        json_data=json.load(f)
        video_results.from_data_dict(json_data)
    min_logit=video_results.box_logits_min()
    print(min_logit)
    max_scorer=max_with_default(min_logit)
    frame_results=video_results.get_frame_scores(max_scorer)
    if len(frame_results)!=1:
        print(f"Warning: only support single label visualization! The results have {len(frame_results)} labels")
    for k in frame_results:
        plt.plot(frame_results[k])
    plt.ylabel("Frame Score")
    plt.xlabel("Frame Index")
    plt.title(f"Frame Scores of {video_results.labels}")
    json_name=file_name(json_path)
    json_name,_=split_suffix(json_name)
    fig_name=f'{json_name}_frame_scores.jpg'
    plt.savefig(fig_name)

def plot_bbox_distribution(json_path):
    with open(json_path,'r') as f:
        results=json.load(f)
    print(results)
    results_np={ name:np.array(logits) for name,logits in results['result'].items()}
    fig, ax = plt.subplots(figsize=(12, 6))
    for name,result in results_np.items():
        ax.plot(result,label=name)
    plt.legend(loc="upper left")
    plt.ylim([0, 35])
    plt.ylabel('Cosine similarity between each language query and  the image')
    plt.xlabel('Frame sampled at 10fps')
    plt.savefig('results.jpg')


def plot_clip_frame_score(result_json):
    with open(result_json,'r') as f:
        data=json.load(f)
    result=data['result']
    fig,ax=plt.subplots(figsize=(8,4))
    for k in result:
        ax.plot(range(0,3*len(result[k]),3),result[k], label = k)
    ax.legend()
    ax.set_xlabel("Frame Index")
    ax.set_ylim(10,40)
    ax.set_xlim(0,2900)
    ax.set_ylabel("Similarity Score")
    plt.savefig('clip_baseline.jpg')
    
    

if __name__=='__main__':
    json_path='data/hong_kong_airport_demo_data.mp4_scores.json'
    plot_frame_score(json_path)