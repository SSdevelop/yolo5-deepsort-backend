from deep_sort_realtime.deepsort_tracker import DeepSort
from outputs import VideoResult
import cv2
import time
import argparse
import json
if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--video_name',type=str)
    parser.add_argument('--result_name',type=str)
    args=parser.parse_args()
    video_result=VideoResult()
    with open(args.result_name) as f:
        results=json.load(f)
    video_result.from_data_dict(results)
    video_input=cv2.VideoCapture(args.video_name)
    track_frame={}
    for label in video_result.labels:
        tracker = DeepSort(max_age=5)
        for frame_result in video_result.frame_results:
            ret,frame=video_input.read()
            if frame_result.skipped():
                continue
            #bbox[2] is the orignal 
            box_frame=[([bbox[2][0], bbox[2][1], bbox[2][2] - bbox[2][0], bbox[2][3]-bbox[2][1]],bbox[0],bbox[3]) for bbox in frame_result.data]
            # print(f'box_frame = {box_frame}')
            if frame is None:
                continue
            start=time.perf_counter()
            tracks = tracker.update_tracks(box_frame, frame=frame) # bbs expected to be a list of detections, each in tuples of ( [left,top,w,h], confidence, detection_class )
            end=time.perf_counter()
            print(f"Find {len(tracks)} tracks in {end-start}s")
            for track in tracks:
                if not track.is_confirmed():
                    continue
                track_id = track.track_id
                print(vars(track))
                if not track_id in track_frame:
                    track_frame[track_id]=[frame_result.frame_index]
                else:
                    track_frame[track_id].append(frame_result.frame_index)
                ltrb = track.to_ltrb()
                print(ltrb)
    print(f'found {len(track_frame)} results')
    with open('track_result.json','w') as f:
        json.dump(track_frame,f)