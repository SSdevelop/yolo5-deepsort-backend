# Open Vocabulary Object Retrieval
## Data Preparation
To prepare the Hong Kong airport video, first download this [video](https://www.youtube.com/watch?v=VRmh2gGeBiE) with `data_provider.py`: `python data_provider.py`; Then, devide it into three minute chunks with ffmpeg: 
```
ffmpeg -i hong_kong_airport.mp4 -c copy -map 0 -segment_time 00:03:00 -f segment -reset_timestamps 1 hong_kong_airport%01d.mp4
```
We have chosen the part between 8:00 to 9:40 as our test data, you can either generate it with ffmpeg, or download
from [google drive link](https://drive.google.com/file/d/1NW670p5VUBKKNl2oZfA80VCNNgit7kcY/view?usp=drive_link)

(Optional) To convert a video into frame with certain fps with the following command
```
ffmpeg -i hong_kong_airport_3.mp4 -vf fps=5 img%04d.jpg
```
## Run Testcases
```
python lf.py --video_name {name} --query_index {index}
```
Here, {name} is the relative path from content root to the video; {index} is the index of the testcase in `test_cases.py`;
For example, to run the first test case:
```
python lf.py --video_name data/hong_kong_airport_demo_data.mp4 --query_index 0
```
The program will load the first testcase which uses the text query `['white backpack','white suitcase','black backpack','black suitcase']`
## Analyzing the Results
`outputs.py` will read the json output by `lf.py` (should be in `results`) and create visualization of top-k frames in a grid
```
python outputs.py --video_results {json output of lf.py } --dump_type {frame|chunk} --top_k {int}
```
For example:
```
python outputs.py --video_results results/hong_kong_airport_demo_data.mp4_202308081040_lang.json --dump_type frame --top_k 50
```
If you want to output chunks:
```
python outputs.py --video_results results/hong_kong_airport_demo_data.mp4_202308081040_lang.json --dump_type frame --top_k 16 --chunk_size 60
```

