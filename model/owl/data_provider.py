from pytube import YouTube
import os
def download(link:str,path,name):
    youtube=YouTube(link)
    video_720p=youtube.streams.get_by_resolution("720p")
    print(f"Download {link} starts")
    video_720p.download(output_path=path,filename=name)

#We are using the video for Hong Kong video
link = [
    #'https://www.youtube.com/watch?v=VRmh2gGeBiE',
    'https://www.youtube.com/watch?v=ZgxirOW9_go'
]
file_names=[
    #'jfk_airport_VRmh2gGeBiE.mp4',
    'hong_kong_airport.mp4',
]


if __name__=='__main__':
    os.makedirs('data',exist_ok=True)
    for i,video_link in enumerate(link):
        download(video_link,'data',file_names[i])

#After downloading the videos, use ffmpeg to devide it into 3 minute chunks (use the ffmpeg command below, ffmpeg should have been installed on cse server)
#https://unix.stackexchange.com/questions/1670/how-can-i-use-ffmpeg-to-split-mpeg-video-into-10-minute-chunks
#ffmpeg -i hong_kong_airport.mp4 -c copy -map 0 -segment_time 00:03:00 -f segment -reset_timestamps 1 hong_kong_airport%01d.mp4