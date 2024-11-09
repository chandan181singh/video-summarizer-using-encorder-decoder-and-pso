import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from path import get_features_path, get_annotation_path
from models import VideoSummarizer, VideoDataset
from eval import evaluate_model
from tqdm import tqdm
from prettytable import PrettyTable

def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=25):
    metrics_table = PrettyTable()
    metrics_table.field_names = ["Epoch", "Loss", "Precision", "Recall", "F1-Score", "Accuracy"]
    
    for epoch in range(num_epochs):
        # Set model to training mode
        model.train()
        total_loss = 0
        
        # Training loop
        for features, labels, mask in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            features, labels, mask = features.to(device), labels.to(device), mask.to(device)
            
            optimizer.zero_grad()
            outputs = model(features)
            
            # Apply mask to both outputs and labels
            outputs = outputs[mask]
            labels = labels[mask]
            
            # Convert labels to binary for training
            binary_labels = (labels > 0.5).float()
            
            loss = criterion(outputs, binary_labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # Calculate average loss
        avg_loss = total_loss/len(train_loader)
        
        # Evaluation phase
        model.eval()  # Set model to evaluation mode
        precision, recall, f1, accuracy = evaluate_model(model, val_loader, device)
        
        # Add metrics to table
        metrics_table.add_row([
            f"{epoch+1}",
            f"{avg_loss:.4f}",
            f"{precision:.4f}",
            f"{recall:.4f}",
            f"{f1:.4f}",
            f"{accuracy:.4f}"
        ])
        
        # Print current epoch's metrics
        if (epoch + 1) % 5 == 0:
            print("\nTraining Metrics:")
            print(metrics_table)
    
    return metrics_table

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize dataset and dataloader
    dataset = VideoDataset(get_features_path(), get_annotation_path())
    
    # Split dataset into train and validation
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
    
    # Initialize model
    model = VideoSummarizer().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Train the model and get metrics
    metrics_table = train_model(model, train_loader, val_loader, criterion, optimizer, device)
    
    # Save the trained model
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics_table.get_string()
    }, 'models/video_summarizer_resnet18_1.pth')

if __name__ == "__main__":
    main()