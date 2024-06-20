# Bone Detection

This repository aims to evaluate the performance of Yolo on pathology recognition in radiography. 

The dataset is sourced from Kaggle. 

We want here to train yolov5s witch is with Roi not with segmentation. 

# Create Datas  createData.py

This module convert segmentation file to Yolo Roi file to be trained by yolov5s. 

The labels of the dataset is initialy with few points. 

This module calculate the bounding rect of these points to create the Region Of Interest in Yolo format. 


# Deep Learning createModel.py. 

createModel.py trains Yolo v5s to detect the pathologies mentioned in datas.yaml. 

It creates the weights file as output. 

# Use model. 

This module reads some images from the dataset and displays the model's predictions. 



