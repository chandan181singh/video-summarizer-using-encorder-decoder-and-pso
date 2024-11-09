import torch
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from models import PSO
import numpy as np

def evaluate_model(model, test_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for features, labels, mask in test_loader:
            features, labels, mask = features.to(device), labels.to(device), mask.to(device)
            outputs = model(features)
            
            # Apply mask to both outputs and labels
            outputs = outputs[mask]
            labels = labels[mask]
            
            # Convert labels to binary (0 or 1)
            binary_labels = (labels > 0.5).cpu().numpy()
            
            # Apply PSO for keyshot selection
            pso = PSO(n_particles=20, n_iterations=50, scores=outputs.cpu().numpy())
            selected_shots = pso.optimize()
            
            all_preds.extend(selected_shots.astype(int))  # Convert boolean to int
            all_labels.extend(binary_labels.astype(int))  # Convert boolean to int
    
    # Convert lists to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Calculate metrics
    precision = precision_score(all_labels, all_preds, average='binary', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='binary', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='binary', zero_division=0)
    accuracy = accuracy_score(all_labels, all_preds)
    
    return precision, recall, f1, accuracy