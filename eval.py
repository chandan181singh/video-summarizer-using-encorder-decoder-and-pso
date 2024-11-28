import torch
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from models import PSO
import numpy as np
from path import DATASET, DATASET_CONFIG

def evaluate_model(model, test_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for features, labels, mask in test_loader:
            features, labels, mask = features.to(device), labels.to(device), mask.to(device)
            outputs = model(features)
            
            outputs = outputs[mask]
            labels = labels[mask]
            
            if DATASET == "SumMe":
                threshold = labels.mean() * 0.7
            else:
                threshold = 0.25
            
            binary_labels = (labels > threshold).cpu().numpy()
            
            pso = PSO(n_particles=50, n_iterations=150, scores=outputs.cpu().numpy())
            selected_shots = pso.optimize()
            
            all_preds.extend(selected_shots.astype(int))
            all_labels.extend(binary_labels.astype(int))
    
    metrics = calculate_metrics(all_preds, all_labels)
    return metrics['precision'], metrics['recall'], metrics['f1'], metrics['accuracy']

def calculate_metrics(all_preds, all_labels):
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    precision = precision_score(all_labels, all_preds, average='binary', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='binary', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='binary', zero_division=0)
    accuracy = accuracy_score(all_labels, all_preds)
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'accuracy': accuracy
    }