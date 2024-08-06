
# Import Libraries

import cv2
import numpy as np
import os
from ultralytics import YOLO
import torch

model = YOLO(r'runs\detect\train3\weights\best.pt')

# Labels  

class_labels = {
    0: 'bus',
    1: 'car',
    2: 'truck',
    3: 'van',
    4: 'long vehicle',
    5: 'others'
}

# Colours

colors = {
    'bus': (0, 255, 0),
    'car': (255, 0, 0),
    'truck': (0, 0, 255),
    'van': (255, 255, 0),
    'long vehicle': (255, 0, 255),
    'others': (0, 255, 255)
}

# Bounding Box and Label To it

def draw_bounding_boxes(frame, detections, counts):
    for detection in detections:
        class_id = detection['class']
        label = class_labels[class_id]
        box = detection['box']
        color = colors[label]

        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 2)
        cv2.putText(frame, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    y_offset = 30
    for label, count in counts.items():
        cv2.putText(frame, f'{label}: {count}', (frame.shape[1] - 200, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.9, colors[label], 2)
        y_offset += 30

# Process the images 

def process_image(image_path, output_path, conf=0.2):
    frame = cv2.imread(image_path)
    results = model.predict(frame, conf=conf, save=False, device=0)  

    detections = []
    counts = {label: 0 for label in class_labels.values()}
    for result in results[0].boxes:
        box = result.xyxy[0].cpu().numpy().astype(int) 
        class_id = int(result.cls[0].cpu().numpy())  
        confidence = result.conf[0].cpu().numpy() 
        if confidence < conf:
            continue  
        label = class_labels[class_id]

        detections.append({'class': class_id, 'box': box})
        counts[label] += 1

    draw_bounding_boxes(frame, detections, counts)
    cv2.imwrite(output_path, frame)
    del frame, results, detections, counts  

# main file making output folder if not and giving the process_image function to make folder path run from model_file path

def main(input_folder, output_folder, conf=0.1):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        file_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)

        if filename.endswith(('.jpg', '.jpeg', '.png')):
            process_image(file_path, output_path, conf)
            torch.cuda.empty_cache() 

if __name__ == '__main__':
    input_folder = r'D:\Users\abhay\Downloads\Car_Count_GE\testing_images\valid\images'
    output_folder = 'output'
    conf_threshold = 0.2  
    main(input_folder, output_folder, conf=conf_threshold)
