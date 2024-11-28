import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from path import FEATURES_PATH, ANNOTATION_PATH, MODEL_NAME, DATASET, ATTENTION_MODEL
from models import VideoSummarizer, VideoDataset
from eval import evaluate_model
from tqdm import tqdm
from prettytable import PrettyTable

def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=40):
    metrics_table = PrettyTable()
    metrics_table.field_names = ["Epoch", "Loss", "Precision", "Recall", "F1-Score", "Accuracy"]
    
    # Increase positive class weight even more
    pos_weight = torch.tensor([12.0 if DATASET == "SumMe" else 10.0]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    # Add learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', 
                                                    factor=0.5, patience=3, 
                                                    verbose=True)
    
    best_f1 = 0
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for features, labels, mask in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            features, labels, mask = features.to(device), labels.to(device), mask.to(device)
            
            optimizer.zero_grad()
            outputs = model(features)
            
            # Apply mask to both outputs and labels
            outputs = outputs[mask]
            labels = labels[mask]
            
            # Dynamic thresholding for binary labels
            if DATASET == "SumMe":
                threshold = labels.mean() + labels.std()
            else:
                threshold = 0.5
            
            binary_labels = (labels > threshold).float()
            
            loss = criterion(outputs, binary_labels)
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            total_loss += loss.item()
        
        # Calculate average loss
        avg_loss = total_loss/len(train_loader)
        
        # Evaluation phase
        model.eval()  # Set model to evaluation mode
        precision, recall, f1, accuracy = evaluate_model(model, val_loader, device)
        
        #print(f"Epoch {epoch+1}: Precision: {precision}, Recall: {recall}, F1: {f1}, Accuracy: {accuracy}")
        # Add metrics to table
        metrics_table.add_row([
            f"{epoch+1}",
            f"{avg_loss:.4f}",
            f"{precision:.4f}",
            f"{recall:.4f}",
            f"{f1:.4f}",
            f"{accuracy:.4f}"
        ])
        
        #Print current epoch's metrics
        if (epoch + 1) == num_epochs:
            print("\nTraining Metrics:")
            print(metrics_table)
        
        # Adjust learning rate based on F1 score
        scheduler.step(f1)
        
        # Save best model
        if f1 > best_f1:
            best_f1 = f1
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'f1': f1
            }, f'{MODEL_NAME}_best')
    
    return metrics_table

def main():
    print(f"Dataset: {DATASET}")
    print(f"Features path: {FEATURES_PATH}")
    print(f"Attention model: {ATTENTION_MODEL}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize dataset and dataloader
    dataset = VideoDataset(FEATURES_PATH, ANNOTATION_PATH)
    
    # Split dataset into train and validation
    # train_size = int(0.8 * len(dataset))
    # val_size = len(dataset) - train_size
    # train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Get video categories from the dataset
    categories = dataset.anno_df['category'].unique()
    
    # Create indices for train and validation splits
    train_indices = []
    val_indices = []
    
    # For each category, take one video for validation and rest for training
    for category in categories:
        # Get indices of videos in this category
        category_indices = [i for i, vid_id in enumerate(dataset.video_ids) 
                          if dataset.anno_df[dataset.anno_df['video_id'] == vid_id]['category'].iloc[0] == category]
        
        # Take one video for validation
        val_indices.append(category_indices.pop())
        # Add remaining videos to training
        train_indices.extend(category_indices)
    
    # Create train and validation datasets using the indices
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
    
    # Initialize model
    model = VideoSummarizer().to(device)
    criterion = nn.BCELoss()  # Binary Cross Entropy Loss
    optimizer = optim.Adam(model.parameters(), lr=0.002, weight_decay=1e-5)
    
    # Train the model and get metrics
    metrics_table = train_model(model, train_loader, val_loader, criterion, optimizer, device)
    
    # Save the trained model
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics_table.get_string()
    }, MODEL_NAME)

if __name__ == "__main__":
    main()