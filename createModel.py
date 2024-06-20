import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import random
from ultralytics import YOLO
import supervision as sv
import torch
import shutil
import argparse
import time


class YoloModel:
    def __init__(self):
        self.model = YOLO('yolov5s.pt')
        self.create_datas()
        self.deep_learning()
        
        
    def create_datas(self):
        yaml_content = f"""
        train: {os.path.join(os.getcwd(), 'YoloDatas/train')}
        test: {os.path.join(os.getcwd(), 'YoloDatas/test')}
        val: {os.path.join(os.getcwd(), 'YoloDatas/valid')}

        nc: 7
        names:
            0: elbow positive
            1: fingers positive
            2: forearm fracture
            3: humerus fracture
            4: humerus
            5: shoulder fracture
            6: wrist positive
        """
        with open('datas.yaml', 'wt') as f:
            f.write(yaml_content) 
            
    def deep_learning(self, epochs=3, datas='datas.yaml'):
        start_time = time.time()
        model = YOLO('yolov5s.pt')
        results = model.train(data=datas, epochs=epochs, imgsz=320)
        result_dir = results.save_dir
        path = os.path.join(os.getcwd(), result_dir) + '/weights/best.pt'
        title = os.path.join(os.getcwd(), 'result_model.pt')
        shutil.move(path, title)
        end_time = time.time()
        duration = end_time - start_time
        duration = round(duration, 2)
        print(f'duation {duration} ms for {epochs} epochs')
        
if __name__=='__main__':
    # parser = argparse.ArgumentParser(description='YOLO Model Training')
    # parser.add_argument('--epochs', type=int, default=3,
    #                     help='number of epochs for training (default: 3)')
    # parser.add_argument('--datas', type=str, default='datas.yaml',
    #                     help='path to the YAML file containing dataset configuration (default: datas.yaml)')
    
    # args = parser.parse_args()

    YoloModel()
    