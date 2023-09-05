import torch
import cv2
import json
from transformers import CLIPProcessor, CLIPModel
#https://huggingface.co/docs/transformers/main/model_doc/owlvit
class Clip:
    def __init__(self) -> None:
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    #note: image-to-text logits transpose is text-to-image logits
    def get_logits(self,image,text_queries):
        inputs=self.processor(text=text_queries, images=image, return_tensors="pt",padding=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
            #we use absolute value as result?
            #probs = logits_per_image.softmax(dim=1) 
        return logits_per_image

    #image feat and text feat still needs to be projected
    def get_image_feat(self,image):
        with torch.no_grad():
            inputs=self.processor(images=image,return_tensors='pt')
            return self.model.get_image_features(**inputs)
        
    #image feat and text feat still needs to be projected 
    def get_text_feat(self,text:str):
        with torch.no_grad():
            inputs=self.processor.tokenizer(text, padding=True, truncation=True, max_length=64, return_tensors='pt')
            return self.model.get_text_features(**inputs)
        


def process_video_clip(model:Clip,video_path:str,text_queries,interval=3,max_frame=None):
    result={s:[] for s in text_queries}
    video=cv2.VideoCapture(video_path)
    frame_count=0
    print(f"Number of frames: {int(video.get(cv2.CAP_PROP_FRAME_COUNT))} Processing interval: {interval}")
    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            print("Ret is false!")
            break
        if max_frame is not None:
            if frame_count>max_frame:
                print(f"Max frame {max_frame} reached.")
                video.release()
                return result
        frame_count=frame_count+1
        if frame_count%interval!=0:
            continue
        logits=model.get_logits(frame,text_queries)
        print(f"Results of frame {frame_count}: {logits}")
        for index in range(len(text_queries)):
            result[text_queries[index]].append(logits[0][index].item())
    video.release()
    return result


if __name__=='__main__':
    #max precision given different threshold criteria
    video_name='./data/hong_kong_airport_demo_data.mp4'
    text_querues=['an image with a black and white backpack in it','a black and white backpack','a image of crowd at a airport']
    # text_queries=[
    #     'checkered tote',
    # ]
    # use owl-vlm
    #process_video_owl(video_name,text_queries,result_dir='hong_kong_airport_3')
    #use clio
    clip=Clip()
    print(text_querues)
    res=process_video_clip(clip,video_name,text_querues,max_frame=2700)
    with open(f'{video_name}_scores.json','w') as f:
        json.dump({'query':text_querues,'result':res},f)