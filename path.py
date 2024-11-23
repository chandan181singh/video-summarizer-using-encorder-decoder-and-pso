import os

DATASET = "TVSum"  # SumMe or TVSum
ATTENTION_MODEL = "Luong"  # Luong or Bahdanau
FEATURE_EXTRACTION_MODEL = "resnet18"

# Base paths for both datasets
if DATASET == "SumMe":
    BASE_PATH = "F:/Video_Summerization/datasets/datasets/SumMe"
    VIDEO_PATH = os.path.join(BASE_PATH, "videos")  # Contains .webm files
    ANNOTATION_PATH = os.path.join(BASE_PATH, "GT")  # Contains .mat files
    VIDEO_EXTENSION = ".webm"
    ANNOTATION_EXTENSION = ".mat"
else:  # TVSum
    BASE_PATH = "F:/Video_Summerization/video-summerization/ydata-tvsum50-v1_1"
    VIDEO_PATH = os.path.join(BASE_PATH, "ydata-tvsum50-video/video")  # Contains video files
    ANNOTATION_PATH = os.path.join(BASE_PATH, "ydata-tvsum50-data/data/ydata-tvsum50-anno.tsv")
    VIDEO_EXTENSION = ".mp4"  # Assuming TVSum videos are in mp4 format
    ANNOTATION_EXTENSION = ".tsv"

# Common paths for both datasets
MODEL_PATH = os.path.join("models")
os.makedirs(MODEL_PATH, exist_ok=True)
MODEL_NAME = os.path.join(MODEL_PATH, f"model_{DATASET}_{FEATURE_EXTRACTION_MODEL}_{ATTENTION_MODEL}_4.pth")

FEATURES_PATH = os.path.join(f"Features/features_{DATASET}_{FEATURE_EXTRACTION_MODEL}_1")
os.makedirs(FEATURES_PATH, exist_ok=True)

SUMMARIZED_VIDEO_PATH = os.path.join("summarized_videos", DATASET)
os.makedirs(SUMMARIZED_VIDEO_PATH, exist_ok=True)

METRICS_PATH = os.path.join(f"metrics/metrics_{DATASET}_{FEATURE_EXTRACTION_MODEL}_{ATTENTION_MODEL}.md")
os.makedirs(os.path.dirname(METRICS_PATH), exist_ok=True)

# Dataset specific configurations
DATASET_CONFIG = {
    "SumMe": {
        "max_summary_length": 0.15,  # 15% of original video length as mentioned in the paper
        "score_range": (0, 1),  # Normalized scores
        "fps": 30  # Default FPS for SumMe videos
    },
    "TVSum": {
        "max_summary_length": 0.15,  # Using same as SumMe for consistency
        "score_range": (1, 5),  # Original score range
        "fps": 30  # Default FPS for TVSum videos
    }
}

# Current dataset configuration
CURRENT_CONFIG = DATASET_CONFIG[DATASET]

