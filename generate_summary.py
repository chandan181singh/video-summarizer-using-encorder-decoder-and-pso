import torch
import numpy as np
import cv2
from models import VideoSummarizer as VSModel
from feature_extraction import extract_features, load_model
import os
from knapsack import knapsack_dp
from kts import cpd_auto, cpd_nonlin
from path import get_model_path, get_random_video_path, get_summarized_video_path
import os
class VideoSummarizer:
    def __init__(self, model_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        # Load the trained model
        self.model = VSModel().to(device)
        checkpoint = torch.load(model_path, map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # Load the feature extraction model
        self.feature_model, _ = load_model()
    
    def temporal_segmentation(self, features, kernel_size=5, num_segments=None):
        """Apply segmentation to divide video into shots"""
        if num_segments is None:
            num_segments = min(len(features) // 30, 20)  # reasonable default
        
        # Force minimum number of segments
        num_segments = max(num_segments, 10)  # Increased minimum segments
        
        # Create segments
        n_frames = len(features)
        segment_size = n_frames // num_segments
        changes = list(range(0, n_frames, segment_size))
        if changes[-1] != n_frames:
            changes.append(n_frames)
        
        return np.array(changes)
    
    def generate_summary(self, video_path, summary_ratio=0.25):
        """Generate video summary"""
        print(f"Processing video: {video_path}")
        
        # Get video info first
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        
        features = extract_features(video_path, self.feature_model, self.device)
        print(f"Features shape: {features.shape}")
        
        # Calculate frame mapping ratio
        frames_per_feature = total_frames / len(features)
        print(f"Frames per feature: {frames_per_feature}")
        
        features_tensor = torch.from_numpy(features).float().to(self.device)
        
        with torch.no_grad():
            scores = self.model(features_tensor.unsqueeze(0)).squeeze().cpu().numpy()
        scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-10)
        
        # Apply temporal segmentation
        segments = self.temporal_segmentation(features)
        print(f"Segments: {segments}")
        
        # Calculate segment scores
        segment_scores = []
        segment_indices = []
        for i in range(len(segments)-1):
            start, end = segments[i], segments[i+1]
            # Convert feature indices to frame indices
            frame_start = int(start * frames_per_feature)
            frame_end = int(end * frames_per_feature)
            segment_score = np.mean(scores[start:end])
            segment_scores.append(segment_score)
            segment_indices.append((frame_start, frame_end))
        
        # Sort segments by score
        sorted_segments = [(score, seg) for score, seg in zip(segment_scores, segment_indices)]
        sorted_segments.sort(reverse=True)
        
        # Select segments to match target duration
        target_frames = int(total_frames * summary_ratio)
        print(f"Target frames: {target_frames}")
        
        selected_segments = []
        current_frames = 0
        
        for _, segment in sorted_segments:
            seg_length = segment[1] - segment[0]
            if current_frames + seg_length <= target_frames:
                selected_segments.append(segment)
                current_frames += seg_length
        
        # Sort by time for smooth playback
        selected_segments.sort(key=lambda x: x[0])
        
        print(f"Selected segments: {selected_segments}")
        print(f"Selected frames: {current_frames} / Target frames: {target_frames}")
        
        # Generate summary video
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        output_filename = os.path.basename(video_path)
        output_basename = os.path.splitext(output_filename)[0] + '_summary3.avi'
        output_path = os.path.join(get_summarized_video_path(), output_basename)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_idx = 0
        included_frames = 0
        
        print("Creating summary video...")
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Check if current frame is in selected segments
            for start, end in selected_segments:
                if start <= frame_idx < end:
                    out.write(frame)
                    included_frames += 1
                    break
                    
            frame_idx += 1
            if frame_idx % 100 == 0:
                print(f"Processed {frame_idx} frames, included {included_frames} frames")
        
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            print(f"Summary saved to: {output_path}")
            print(f"Original frames: {frame_idx}, Summary frames: {included_frames}")
            return output_path
        else:
            raise RuntimeError("Failed to create summary video")

def main():
    
    model_path = os.path.join(get_model_path(), "video_summarizer.pth")
    print(model_path)

    summary_path = os.path.join(get_random_video_path(), "tyre_change.mp4")
    print(summary_path)

    summarizer = VideoSummarizer(model_path)
    summary_path = summarizer.generate_summary(
        summary_path, 
        summary_ratio=0.25
    )
    
if __name__ == "__main__":
    main() 