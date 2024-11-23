import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import cv2
import os
import numpy as np
from tqdm import tqdm
from path import VIDEO_PATH, FEATURES_PATH, FEATURE_EXTRACTION_MODEL

def load_model():
    # Load pre-trained ResNet model
    if FEATURE_EXTRACTION_MODEL=="resnet18":
        model = models.resnet18(weights='IMAGENET1K_V1')
    else:
        model = models.resnet50(weights='IMAGENET1K_V1')
    
    # Remove the final fully connected layer
    model = nn.Sequential(*list(model.children())[:-1])
    model.eval()
    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    return model, device

def preprocess_frame(frame):
    # Define the same transforms used during training
    transform = transforms.Compose([
        transforms.ToPILImage(),  # Convert numpy array (OpenCV) to PIL Image
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform(frame)

def extract_features(video_path, model, device, sample_rate=30):
    cap = cv2.VideoCapture(video_path)
    features = []
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Process every nth frame based on sample_rate
        if frame_count % sample_rate == 0:
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Preprocess the frame
            input_tensor = preprocess_frame(frame)
            input_batch = input_tensor.unsqueeze(0).to(device) # Add batch dimension at index 0
            
            # Extract features
            with torch.no_grad():
                feature = model(input_batch)
                feature = feature.squeeze().cpu().numpy()
                features.append(feature)
                
        frame_count += 1
    
    cap.release()
    return np.array(features)

def main():
    # Directory containing videos
    video_dir = VIDEO_PATH  # Change this to your video directory
    output_dir = FEATURES_PATH  # Directory to save features
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the model
    model, device = load_model()
    print(f"Using device: {device}")
    
    # Process each video in the directory
    video_files = [f for f in os.listdir(video_dir) if f.endswith(('.mp4', '.avi', '.mov'))]
    
    for video_file in tqdm(video_files, desc="Processing videos"):
        # Get video ID from filename (without extension)
        video_id = os.path.splitext(video_file)[0]
        video_path = os.path.join(video_dir, video_file)
        
        # Extract features
        features = extract_features(video_path, model, device)
        
        # Save features with consistent naming
        output_path = os.path.join(output_dir, f"{video_id}_features.npy")
        np.save(output_path, features)
        
        print(f"Processed {video_id}: Feature shape {features.shape}")

if __name__ == "__main__":
    main()

