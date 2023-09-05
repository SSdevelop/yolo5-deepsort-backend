from transformers import pipeline
import numpy as np
from PIL import Image
from PIL import ImageDraw
import cv2 
import os
import argparse
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
import torch
import torchvision
#nms per class!

def inverse_sigmoid(scores):
    return torch.log(scores/(torch.ones_like(scores,dtype=torch.float)-scores))

class FrameProcessor:
    def __init__(self) -> None:
        checkpoint = "google/owlvit-base-patch32"
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(checkpoint)
        self.processor = AutoProcessor.from_pretrained(checkpoint)

    def process_image(self,image,text_queries,device='cpu'):
        inputs=self.processor(text=text_queries, images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            target_sizes=[(image.shape[0],image.shape[1])]
            #out is OwlViTObjectDetectionOutput, need to move back to cpu
            outputs.logits=outputs.logits.to('cpu') 
            outputs.pred_boxes=outputs.pred_boxes.to('cpu')
            raw_results = self.processor.post_process_object_detection(outputs, threshold=0.05,target_sizes=target_sizes)
            #raw result length=1 since only one frame
            assert(len(raw_results)==1)
        keep_indexs=torchvision.ops.batched_nms(raw_results[0]["boxes"],raw_results[0]["scores"],raw_results[0]['labels'],0.3)
        results={
            'scores':raw_results[0]["scores"][keep_indexs].tolist(),
            'labels':raw_results[0]["labels"][keep_indexs].tolist(),
            'logits':inverse_sigmoid(raw_results[0]["scores"][keep_indexs]).tolist(),
            'boxes':raw_results[0]["boxes"][keep_indexs].tolist()
        }
        return results
    def image_query(self,image,image_query,device='cpu'):
        #This processor will resize both frame and qury image to 768*768
        #We do not want this since it will squash image of a person to a square
        #However, if we set do_resize=False,do_center_crop=False, we need to change model config as well
        inputs = self.processor(images=image, query_images=image_query, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = self.model.image_guided_detection(**inputs)
        # Target image sizes (height, width) to rescale box predictions [batch_size, 2]
        target_sizes = [[image.shape[0],image.shape[1]]]
        target_sizes = torch.Tensor(target_sizes)
        # Convert outputs (bounding boxes and class logits) to COCO API
        #print(outputs)
        raw_results = self.processor.post_process_image_guided_detection(
            outputs=outputs, threshold=0.8, nms_threshold=0.3, target_sizes=target_sizes
        )
        assert(len(raw_results)==1)
        #no label for image query
        results={
            'scores':raw_results[0]["scores"].tolist(),
            'labels':torch.zeros_like(raw_results[0]["scores"],dtype=torch.int).tolist(),
            'boxes':raw_results[0]["boxes"].tolist(),
            'logits':inverse_sigmoid(raw_results[0]["scores"]).tolist(),
        }
        return results

def visualize_results(image,result,classes,top_left_caption):
    image=cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
    #different color for different bounding box
    color_step=255//len(classes)
    colors=[(255-i*color_step,0,i*color_step) for i in range(1,len(classes)+1)]
    cv2.putText(image, top_left_caption,(5,25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2)
    for index,label in enumerate(result['labels']):
        point=result['boxes'][index]
        p1_x,p1_y,p2_x,p2_y=int(point[0]),int(point[1]),int(point[2]),int(point[3])
        #print(f"P1： ({p1_x},{p1_y}) P2: ({p2_x},{p2_y})")
        cv2.rectangle(img=image,pt1=(p1_x,p1_y),pt2=(p2_x,p2_y),color=colors[label],thickness=3)
        score_str="{:.2f}".format(result['scores'][index])
        cv2.putText(image, f"{label+1} ({score_str})",(p1_x+5, p1_y+25), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    return image

#singleton object
vlm_processor=FrameProcessor()

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--img",type=str,default='./data/SmartCity/images/39.jpg')
    args = parser.parse_args()
    checkpoint = "google/owlvit-base-patch32"
    model = AutoModelForZeroShotObjectDetection.from_pretrained(checkpoint)
    processor = AutoProcessor.from_pretrained(checkpoint)
    image = cv2.imread(args.img)
    #print(f"Image shape: {image.shape}")
    #text_queries=["human face", "rocket", "nasa badge", "star-spangled banner"]
    text_queries=['backpack','lift','girl in pink']
    inputs=processor(text=text_queries, images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        target_sizes=[(image.shape[0],image.shape[1])]
        raw_results = processor.post_process_object_detection(outputs, threshold=0.1,target_sizes=target_sizes)[0]
    results={
        'scores':raw_results["scores"].tolist(),
        'labels':raw_results["labels"].tolist(),
        'boxes':raw_results["boxes"].tolist()
    }
    visualize_image=visualize_result_image(image,results,text_queries)
    cv2.imwrite(f"{str(os.path.basename(os.path.join(args.img)))}_process.jpg",visualize_image)