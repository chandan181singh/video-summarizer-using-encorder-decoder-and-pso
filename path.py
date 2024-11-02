import os

# Base path for the dataset
BASE_PATH = "F:/Video_Summerization/video-summerization/ydata-tvsum50-v1_1"

# Ensure the base path exists
if not os.path.exists(BASE_PATH):
    raise FileNotFoundError(f"Base path {BASE_PATH} does not exist")

# Define paths for different components
MODEL_PATH = os.path.join("models")
FEATURES_PATH = os.path.join("features_resnet18")
VIDEO_PATH = os.path.join(BASE_PATH, "ydata-tvsum50-video/video")
ANNOTATION_PATH = os.path.join(BASE_PATH, "ydata-tvsum50-data/data/ydata-tvsum50-anno.tsv")
RANDOM_VIDEO_PATH = os.path.join("random_video")
SUMMARIZED_VIDEO_PATH = os.path.join("summarized_videos")

# Create directories if they don't exist
os.makedirs(MODEL_PATH, exist_ok=True)
os.makedirs(FEATURES_PATH, exist_ok=True)
os.makedirs(SUMMARIZED_VIDEO_PATH, exist_ok=True)

# Function to get paths
def get_model_path():
    return MODEL_PATH

def get_features_path():
    return FEATURES_PATH

def get_video_path():
    return VIDEO_PATH

def get_base_path():
    return BASE_PATH

def get_annotation_path():
    return ANNOTATION_PATH

def get_random_video_path():
    return RANDOM_VIDEO_PATH

def get_summarized_video_path():
    return SUMMARIZED_VIDEO_PATH
