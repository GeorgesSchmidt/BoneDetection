import cv2
import numpy as np
import os
import argparse
import shutil
import re

class ReadData:
    def __init__(self, folder='YoloDatas') -> None:
        parent_directory = os.path.join(os.getcwd(), 'YoloDatas')
        self.names = ['no patho', 'elbow positive', 'fingers positive', 'forearm fracture', 'humerus fracture', 'humerus', 'shoulder fracture', 'wrist positive']
 
        self.get_paths(parent_directory)
        self.get_images()
        
        
    def get_paths(self, parent_directory):
        directories = ['train', 'test', 'valid']
        self.path_images, self.path_labels = [], []
        for directory in directories:
            paths = os.listdir(os.path.join(parent_directory, directory))
            for path in paths:
                if path.endswith('.png'):
                    path = os.path.join(parent_directory, directory, path)
                    self.path_images.append(path)
                if path.endswith('.txt'):
                    path = os.path.join(parent_directory, directory, path)
                    self.path_labels.append(path)
                    
        self.path_images = sorted(self.path_images, key=lambda s: int(re.search(r'\d+', s).group()))
        self.path_labels = sorted(self.path_labels, key=lambda s: int(re.search(r'\d+', s).group()))
        
    def get_images(self):
        self.arr_images, self.arr_labels = [], []
        for path_img, path_label in zip(self.path_images[:20], self.path_labels[:20]):
            title = path_img.split('/')[-1]
            title = title.split('_')[1]
            
            img = cv2.imread(path_img)
            datas = self.read_label(path_label, img)
            for num, x0, y0, x1, y1 in datas:
                name = self.names[num]
                cv2.rectangle(img, (x0, y0), (x1, y1), (0, 255, 0), 2)
                p = (x0, y0-10)
                self.put_text(img, name, p, (0, 255, 0))
            cv2.imshow('', img)
            cv2.waitKey(0)
                    
                    
    def read_label(self, path, img):
        h, w = img.shape[:2]
        with open(path, 'r') as f:
            content = f.readlines()
        result = []
        for line in content:
            num = line.split(':')[0]
            
            sub = line.split(' ')
            c_x, c_y, width, height =  map(float, sub[1:])
            x0 = int((c_x - (width*0.5))*w)
            y0 = int((c_y - (height*0.5))*h)
            
            x1 = int((c_x + (width*0.5))*w)
            y1 = int((c_y + (height*0.5))*h)
            result.append([int(num), x0, y0, x1, y1])
            
        return result
            
    def put_text(self, frame, texte, p, color=(0, 0, 0)):
        """
        Add text to a frame.
        
        Args:
            frame (numpy.ndarray): The frame from the video.
            texte (str): The text to add.
            p (tuple): The position to add the text.
            color (tuple): The color of the text.
        """
        police = cv2.FONT_HERSHEY_SIMPLEX  # Choix de la police
        taille_police = 1  # Taille du texte
        epaisseur = 2  # Épaisseur du texte

        # Ajouter le texte sur l'image
        cv2.putText(frame, texte, p, police, taille_police, color, epaisseur, cv2.LINE_AA)

            
            
            
    def read_label1(self, path):
        """
        Read labels from a text file and parse them into a structured format.
        
        Args:
        - path (str): Path to the label text file.
        
        Returns:
        - labels (list): List of labels.
        - points (list): List of lists, where each inner list contains points.
        """
        with open(path, 'r') as f:
            content = f.readlines()

        points, labels = [], []
        for line in content:
            sub = line.split()
            num = int(sub[0])
            values = [float(value) for value in sub[1:]]
            pts = [(values[i], values[i + 1]) for i in range(0, len(values), 2)]
            labels.append(num)
            points.append(pts)
        
        return labels, points

                    
    
                    
if __name__=='__main__':
    ReadData()