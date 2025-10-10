import os
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import v2

# preprocessing dataset to feed images and crops
class LRDDDataset(Dataset):
    def __init__(self, csv_file, img_folder, labels_folder):
        self.metadata = pd.read_csv(csv_file)
        self.imgs = img_folder
        self.yolo_labels = labels_folder
        self.transform = v2.Compose([
            v2.Resize((224,224)), 
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale = True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        img_name = row['img_name']
        
        img = os.path.join(self.imgs, row['img_name'])
        
        img = Image.open(img).convert('RGB')
        
        img_width,img_height = img.size
        
        label_path = os.path.join(self.yolo_labels, img_name.replace('.jpg', '.txt'))
        
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                line = f.readline().strip()
                if not line:
                    crop = img
                    width = height = area = aspect = 0.0
                else: 
                    line = line.split()
                    _, x_center, y_center, width, height = map(float, line)
                    #calculating the crop coordinates to get drone crops from image
                    x_center_pixels, y_center_pixels, width_pixels, height_pixels = x_center*img_width, y_center*img_height, width*img_width, height*img_height 
                    
                    x_min, y_min, x_max, y_max = x_center_pixels-width_pixels/2, y_center_pixels-height_pixels/2, x_center_pixels+width_pixels/2, y_center_pixels+height_pixels/2
                    
                    crop = img.crop((x_min, y_min, x_max, y_max))
                    
                    # calculating bounding box features: 
                    area = width * height
                    aspect = width/(height + 1e-6)
        else: 
            crop = img
            width = height = area = aspect = 0.0
        
        img_tensor = self.transform(img)
        crop_tensor = self.transform(crop)
        distance = torch.tensor(row['distance_3d_ft'], dtype=torch.float32)
        bbox_features = torch.tensor([width, height, area, aspect], dtype=torch.float32)
        
        return img_tensor, crop_tensor, bbox_features, distance
