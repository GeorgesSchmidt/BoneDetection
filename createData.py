import cv2
import numpy as np
import os
import argparse
import shutil

class CreateData:
    def __init__(self, directory_input, directory_output) -> None:
        """
        Initialize the CreateData class with input and output directories.
        
        Args:
        - directory_input (str): Directory containing input data (train, test, valid).
        - directory_output (str): Directory to output processed data.
        """
        self.names = ['no patho', 'elbow positive', 'fingers positive', 'forearm fracture', 'humerus fracture', 'humerus', 'shoulder fracture', 'wrist positive']
        self.directory_input = directory_input
        self.directory_output = directory_output
        
        # Check and create necessary directories
        self.check_and_create_directories()
        
        self.read_folder(input_folder=os.path.join(os.getcwd(), 'Datas', 'train'))
        self.read_folder(input_folder=os.path.join(os.getcwd(), 'Datas', 'test'))
        self.read_folder(input_folder=os.path.join(os.getcwd(), 'Datas', 'valid'))
         
    def check_and_create_directories(self):
        """
        Check if output directories exist and clear YoloDatas directory if it exists and is not empty.
        """
        directories = ['train', 'test', 'valid']
        for directory in directories:
            paths = os.listdir(os.path.join(os.getcwd(), 'YoloDatas', directory))
            for path in paths:
                path = os.path.join(os.getcwd(), 'YoloDatas', directory, path)
                os.remove(path)

    def read_folder(self, input_folder):
        print(input_folder)
        direct = input_folder.split('/')[-1]
        paths_img = os.listdir(os.path.join(input_folder, 'images'))
        dic_img = {}
        for path in paths_img:
            key = path.split('.')[0]
            path = os.path.join(input_folder, 'images', path)
            img = cv2.imread(path)
            dic_img[key] = img
        
        paths_labels = os.listdir(os.path.join(input_folder, 'labels'))
        dic_label = {}
        for path in paths_labels:
            key = path.split('.')[0]
            path = os.path.join(input_folder, 'labels', path)
            datas = self.read_label(path)
            dic_label[key] = datas
            
        n=0
        for key, img in dic_img.items():
            nums, points = dic_label[key]
            h, w = img.shape[:2]
            arr = []
            if len(points) == 0:
                name = self.names[0]
                arr.append([0, 0, 0, 0, 0])
                
            for num, pts in zip(nums, points):
                name = self.names[num+1]
                line = []
                for p in pts:
                    x, y = p
                    p = np.array([x*w, y*h]).astype(int)
                    line.append(p)
                line = np.array(line)
                x, y, width, height = cv2.boundingRect(line)
                c_x = (x+(width*0.5))/w
                c_y = (y+(height*0.5))/h
                width /= w
                height /= h
                arr.append([num+1, c_x, c_y, width, height])
                
            title = os.path.join(os.getcwd(), 'YoloDatas', direct, f'img_{name}_{n}.png')
            cv2.imwrite(title, img)
            title = os.path.join(os.getcwd(), 'YoloDatas', direct, f'img_{name}_{n}.txt')
            self.write_file(arr, title)
            
            n += 1
                        
        
    def read_folder1(self, input_folder):
        """
        Read images and labels from a specified folder path and process them.
        
        Args:
        - path (str): Path to the folder containing images and labels subdirectories.
        """
        
        
        direct = input_folder.split('/')[-1]
        print(direct)
        paths_images = os.listdir(os.path.join(input_folder, 'images'))
        paths_labels = os.listdir(os.path.join(input_folder, 'labels'))
        
        self.dic_img, self.dic_label = {}, {}
        
        for title in paths_images:
            name = title.split('.')[0]
            if name not in self.dic_img:
                path = os.path.join(os.getcwd(), 'Datas', 'train', 'images', title)
                self.dic_img[name] = path
                
        for title in paths_labels:
            name = title.split('.')[0]
            if name not in self.dic_label:
                path = os.path.join(os.getcwd(), 'Datas', 'train', 'labels', title)
                self.dic_label[name] = path
    
        n = 0
        for key, path_img in self.dic_img.items():
            img = cv2.imread(path_img)
            h, w = img.shape[:2]
            path_label = self.dic_label[key]
            num, points = self.read_label(path_label)
            name = self.names[0]
            path_img = os.path.join(os.getcwd(), 'YoloDatas', direct, f'{name}_{n}.png')
            cv2.imwrite(path_img, img)
            
            p = (10, 10)
            if len(points)==0:
                self.put_text(img, name, p, (0, 255, 0))
            line = []
            arr = []
            if len(points) == 0:
                arr.append([0, 0, 0, 0, 0])
            for num, pts in zip(num, points):
                name = self.names[num+1]
                line = []
                for p in pts:
                    x, y = p
                    p = np.array([x*w, y*h]).astype(int)
                    line.append(p)
                line = np.array(line)
                #cv2.polylines(img, [line], True, (0, 255, 0), 2)
                x, y, width, height = cv2.boundingRect(line)
                #cv2.rectangle(img, (x, y), (x+width, y+height), (0, 0, 255), 2)
                p = (x, y-10)
                #self.put_text(img, name, p, (0, 255, 0))
                c_x = (x+(width*0.5))/w
                c_y = (y+(height*0.5))/h
                width /= w
                height /= h
                arr.append([num+1, c_x, c_y, width, height])
                
            
            
            path_label = os.path.join(os.getcwd(), 'YoloDatas', direct, f'{name}_{n}.txt')
            self.write_file(arr, path_label)
            # cv2.imshow('', img)
            # cv2.waitKey(0)
            n+=1
        

    
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

    def write_file(self, arr, filename):
        """
        Write data array to a text file.
        
        Args:
        - arr (list): List of lists, where each inner list contains data to write.
        - filename (str): Name of the file to write.
        """
        with open(filename, 'w') as f:
            for row in arr:
                num = row[0]
                line = f'{num}:'
                for v in row[1:]:
                    line += f' {v}'
                line += '\n'
                f.write(line)
                
        
    def read_label(self, path):
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

def parse_arguments():
    """
    Parse command-line arguments.
    
    Returns:
    - args (argparse.Namespace): Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Process input and output directories for image and label processing.")
    parser.add_argument('--input', '-i', type=str, default='Datas', help='Directory containing input data (train, test, valid). Default: Datas')
    parser.add_argument('--output', '-o', type=str, default='YoloDatas', help='Directory to output processed data. Default: OwnData')
    return parser.parse_args()

if __name__ == '__main__':
    # Parse command-line arguments
    args = parse_arguments()
    
    # Initialize CreateData instance with parsed arguments
    CreateData(args.input, args.output)
