import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import random
from ultralytics import YOLO
import supervision as sv

class UseModel:
    def __init__(self, model) -> None:
        self.model = YOLO(model)
        self.bounding_box_annotator = sv.BoundingBoxAnnotator()
        self.get_images()
        #self.show()
        
    def get_images(self):
        path = os.path.join('Datas', 'train', 'images')
        paths = os.listdir(path)
        self.arr_images, self.arr_pred = [], []
        for p in paths:
            if p.endswith('.jpg'):
                title = os.path.join('Datas', 'train', 'images', p)
                img = cv2.imread(title)
                if img is not None:
                    pred = self.get_pred(img)
                    if len(pred) > 0:
                        self.arr_pred.append(pred)
                        self.arr_images.append(img)
                    
        print('images', len(self.arr_images), len(self.arr_pred))
        
                    
    def show(self):
        image = None
        for img in self.arr_images:
            pred = self.get_pred(img)
            if len(pred) > 0:
                print('pred', len(pred))
                image = img
                break
        print('image', image.shape)
            
            
    def get_pred(self, frame):
        results = self.model(frame)[0]
        detections = sv.Detections.from_ultralytics(results)
        return detections
       
                
                
                
        
        
if __name__=='__main__':
    model = 'result_model.pt'
    UseModel(model)