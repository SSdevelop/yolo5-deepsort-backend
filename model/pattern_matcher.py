import numpy as np
import time
from outputs import FrameResult
import json
import cv2


#inputs:
#bbox in (x_0,y_0,x_1,y_1) format 
#frame_boxes: (len(frame),5)
#transformed_boxes: (len(pattern),5,len(parem))
#returns: (len(pattern),len(frame),len(parem)) i.e. an iou matrix for each parem combination
#reference: https://stackoverflow.com/questions/57897578/efficient-way-to-calculate-all-ious-of-two-lists
def _iou(frame_boxes,transformed_boxes):
    low = np.s_[:,:,(0,1),:]
    high = np.s_[:,:,(2,3),:]
    frame_boxes=np.expand_dims(frame_boxes,(0,3))
    transformed_boxes=np.expand_dims(transformed_boxes,(1))
    A,B = frame_boxes.copy(),transformed_boxes.copy()
    A[high] += 1 
    B[high] += 1
    #make sure the boxes are in xyxy format
    assert(np.all(A[high]>A[low]))
    assert(np.all(B[high]>B[low]))
    intersection_area = np.maximum(0,np.minimum(A[high],B[high])-np.maximum(A[low],B[low])).prod(2)
    a_area=(A[high]-A[low]).prod(2)
    b_area=(B[high]-B[low]).prod(2)
    union_area=a_area+b_area-intersection_area
    ious=intersection_area / union_area
    #sanity check: iou<1
    assert(np.all(ious<=1))
    return ious

#Generate the grid for grid search for best offset and scaling factor for the pattern
class ParemGrid:
    def __init__(self,frame_w=1280,frame_h=720,sampling_rate=50,scale_range=(0.125,1.5,0.125)) -> None:
        parem_grid_raw=np.mgrid[0:frame_w:sampling_rate,0:frame_h:sampling_rate,scale_range[0]:scale_range[1]:scale_range[2]]
        self.parem_grid=np.reshape(parem_grid_raw,(3,-1),'F') #F stands for Fortran style row major ordering
        self.len=self.parem_grid.shape[1]
    def xy(self):
        return self.parem_grid[(0,1),:]
    def k(self):
        return self.parem_grid[(2),:]

    def __len__(self):
        return self.len

class Pattern:
    
    def __init__(self,patter_list) -> None:
        self.data=patter_list
        self._np=np.array(patter_list)
        print(f"Pattern Array: {self._np}")
        self.len=self._np.shape[0]

    def xywh(self):
        return self._np[:,[0,1,2,3]]
    def xyxy(self):
        return FrameResultNp.xywh2xyxy(self.xywh())

    def classes(self):
        return self._np[:,4]

    #to preserve the shape, we must transform with xyxy format
    def transform_with_parem(self,parem:ParemGrid):
        xyxy=self.xyxy()
        lower_point=np.expand_dims(xyxy[:,(0,1)],(2))
        higher_point=np.expand_dims(xyxy[:,(2,3)],(2))
        classes=np.expand_dims(self.classes(),(1,2))
        k=np.expand_dims(parem.k(),(0,1))
        x_0_y_0=np.expand_dims(parem.xy(),(0))
        lower_scaled,higher_scaled=k*lower_point,k*higher_point
        lower_shifted=lower_scaled+x_0_y_0
        high_shifted=higher_scaled+x_0_y_0
        classes_broadcasted=np.broadcast_to(classes,(classes.shape[0],classes.shape[1],len(parem)))
        new_boxes=np.concatenate([lower_shifted,high_shifted,classes_broadcasted],(1))
        return new_boxes
    
    def __len__(self):
        return self.len

    def visualize(self):
        image=cv2.imread('misc/frame_550.jpg')
        pattern_box_xyxy=FrameResultNp.xywh2xyxy(self._np).tolist()
        #different color for different bounding box
        for index,point in enumerate(pattern_box_xyxy):
            p1_x,p1_y,p2_x,p2_y=int(point[0]),int(point[1]),int(point[2]),int(point[3])
            #print(f"P1： ({p1_x},{p1_y}) P2: ({p2_x},{p2_y})")
            cv2.rectangle(image,pt1=(p1_x,p1_y),pt2=(p2_x,p2_y),color=(0,255,0),thickness=3)
            cv2.putText(image, f"{index}",(p1_x+5, p1_y+25), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        cv2.imwrite("pattern_visualized.jpg",image)


#A wrapper class for FrameResult class which provides numpy functionalities
class FrameResultNp:
    @staticmethod
    def xyxy2xywh(xyxy):
        w=xyxy[:,2]-xyxy[:,0]
        w=np.expand_dims(w,axis=1)
        h=xyxy[:,3]-xyxy[:,1]
        h=np.expand_dims(h,axis=1)
        return np.concatenate([xyxy[:,(0,1)],w,h],axis=1)
    @staticmethod
    def xywh2xyxy(xywh):
        x=xywh[:,0]+xywh[:,2]
        x=np.expand_dims(x,axis=1)
        y=xywh[:,1]+xywh[:,3]
        y=np.expand_dims(y,axis=1)
        return np.concatenate([xywh[:,(0,1)],x,y],axis=1)
    
    def __init__(self,frame_result:FrameResult):
        self.data=frame_result.data
        numpy_arr=[]
        for record in self.data:
            score,logit,box,label=record
            numpy_arr.append([score,logit,*box,label])
        self._np=np.array(numpy_arr)
    def logits(self):
        pass
    def scores(self):
        pass
    def boxes_xywh(self):
        xyxy=self.boxes_xyxy()
        return self.xyxy2xywh(xyxy)
    def boxes_xyxy(self):
        return self._np[:,2:6]
    def box_xyxyc(self):
        return self._np[:,2:7]
    def boxes_classes(self):
        return self._np[:,6]
    def get_boxes(self,index_loc):
         return self._np[index_loc,:] 

#return max_loc: the index of parem grid where average iou is maximized
#return matched_box_index: index of frame boxes which most match with the pattern
#len(match_box_index)==len(pattern)
def match_pattern_impl(frame_result_boxes,transformed_parem_boxes):
    #iou result: len(pattern)*len(frame)*len(parem)
    iou_frame_pattern=_iou(frame_result_boxes,transformed_parem_boxes)
    #set iou to zero for all classes that do not match
    frame_box_classes=np.expand_dims(frame_result_boxes[:,4],(0,2))
    #broadcast to match dimension
    transformed_parem_boxes_classes=np.expand_dims(transformed_parem_boxes[:,4],(1))
    class_not_eq=frame_box_classes!=transformed_parem_boxes_classes
    class_matched_iou=np.where(class_not_eq,0,iou_frame_pattern)
    # use maximum of each box to assign
    max_index=np.argmax(class_matched_iou,axis=1)
    index_other_dim_1=np.tile(np.arange(max_index.shape[0]),(max_index.shape[1],1)).T
    index_other_dim_2=np.tile(np.arange(max_index.shape[1]),(max_index.shape[0],1))
    max_box_seleced=iou_frame_pattern[index_other_dim_1,max_index,index_other_dim_2]
    score_parem=np.average(max_box_seleced,axis=0)
    # print(f"Scores of parem: {score_parem} shape: {score_parem.shape}")
    max_loc=np.argmax(score_parem)
    matched_box_index=max_index[:,max_loc]
    best_ious=iou_frame_pattern[:,matched_box_index,max_loc]
    return max_loc,matched_box_index,score_parem[max_loc],best_ious


class PatternScorer:
    def __init__(self,pattern,frame_w=1280,frame_h=720,sampling_step=50,scale_range=(0.125,1.5,0.125),class_list=None) -> None:
        self.class_list=class_list
        if self.class_list is not None:
            color_step=255//len(self.class_list)
            colors=[(255-i*color_step,0,i*color_step) for i in range(1,len(self.class_list)+1)]
            self.class_color=colors
        else:
            self.class_color=None
        self.pattern=Pattern(patter_list=pattern)
        self.frame_width=frame_w
        self.frame_height=frame_h
        self.sample_step=sampling_step
        self.scale_range=scale_range

    
    
    def score(self,frame_result:FrameResult):
        frame_result_np=FrameResultNp(frame_result)
        parem_grid=ParemGrid(frame_h=self.frame_height,frame_w=self.frame_width,sampling_rate=self.sample_step,scale_range=self.scale_range)
        transformed_boxes=self.pattern.transform_with_parem(parem_grid)
        max_parem_loc,match_box_loc,avg_iou,best_ious=match_pattern_impl(frame_result_np.box_xyxyc(),transformed_boxes)
        best_boxes=transformed_boxes[:,:,max_parem_loc].tolist()
        result={
            # scale and shifted pattern bounding box which maximize avg_iou with boxes in the frame
            'pattern_box':best_boxes,
            # the index of the boxes in frame_result which matches the pattern
            # use frame_result.data[match_box_loc[i]] to get the corresponding box
            'match_box_index':match_box_loc,
            #average iou of matched box and pattern box
            'avg_iou':avg_iou,
            #iou of 'pattern_box'and matched box in the frame
            'pattern_box_iou':best_ious,
        }
        return result

    #return average iou weighted by confidence score
    #the confidence score better be scaled between [0,1]
    def score_weighted(self):
        pass

    def visualize_frame_result(self,frame_result:FrameResult)->None:
        labels=frame_result.get_prop('labels')
        boxes=frame_result.get_prop('boxes')
        scores=frame_result.get_prop('scores')
        for index,label in enumerate(labels):
            point=boxes[index]
            p1_x,p1_y,p2_x,p2_y=int(point[0]),int(point[1]),int(point[2]),int(point[3])
            #print(f"P1： ({p1_x},{p1_y}) P2: ({p2_x},{p2_y})")
            cv2.rectangle(img=image,pt1=(p1_x,p1_y),pt2=(p2_x,p2_y),color=self.class_color[label],thickness=3)
            #score_str="{:.2f}".format(scores[index])
            #cv2.putText(image, f"{label+1} ({score_str})",(p1_x+5, p1_y+25), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

    #pattern_boxes: xyxyc format ('pattern_box' of result dict of score)
    def visualize_boxes(self,pattern_boxes):
        for point in pattern_boxes:
            p1_x,p1_y,p2_x,p2_y=int(point[0]),int(point[1]),int(point[2]),int(point[3])
            cv2.rectangle(image,pt1=(p1_x,p1_y),pt2=(p2_x,p2_y),color=(0,255,0),thickness=3)
            cv2.putText(image, f"{int(point[4])}",(p1_x+5, p1_y+25), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)


if __name__=="__main__":
    #use frame 550 of 100s video as test
    parem_grid=ParemGrid()
    #Person with two luggage pattern xywh
    #the class index of the pattern must be the same as frame result
    #here, index of person is 2; index of luggage is 3
    pattern_man_two_luggages=[
        (0,100,50,100,3), #luggage one
        (50,0,75,200,2), #person
        (75+50,100,50,100,3), #luggage two
    ]
    pattern_scorer=PatternScorer(pattern=pattern_man_two_luggages,class_list=["white shirt", "blue jeans", "white shirt blue jeans man", "black suitcase"])
    frame_result=FrameResult()
    with open('misc/pattern_test.json') as f:
        data=json.load(f)
        frame_result.from_data_dict(data)
    start=time.perf_counter()
    result_dict=pattern_scorer.score(frame_result)
    print(result_dict)
    end=time.perf_counter()
    image=cv2.imread("misc/frame_550.jpg")
    #visualize the boxes from the frame
    pattern_scorer.visualize_frame_result(frame_result)
    #visualize the match pattern boxes
    best_boxes=result_dict['pattern_box']
    pattern_scorer.visualize_boxes(best_boxes)
    cv2.imwrite("result_550_visualized.jpg",image)
    print(f"Run time: {end-start}s")
    



