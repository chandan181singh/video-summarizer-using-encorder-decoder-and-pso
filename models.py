import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset
import os
import pandas as pd
import math
from luang_attention import LuongAttention

class VideoSummarizer(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=256, num_layers=2, dropout=0.1):
        super(VideoSummarizer, self).__init__()
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Encoder (BiLSTM)
        self.encoder = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Decoder (LSTM)
        self.decoder = nn.LSTM(
            input_size=hidden_dim * 2,  # Double size due to bidirectional encoder
            hidden_size=hidden_dim * 2,  # Match encoder output size
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Attention mechanism
        self.attention = LuongAttention(hidden_dim * 2, method='general')  # Double size to match encoder output
        
        # Output layer
        self.output_layer = nn.Linear(hidden_dim * 2, 1)
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # Project input
        x = self.input_proj(x)  # [batch_size, seq_len, hidden_dim]
        
        # Encode
        encoder_outputs, (hidden, cell) = self.encoder(x)
        
        # Prepare decoder initial states (combine bidirectional states)
        hidden = hidden.view(self.encoder.num_layers, 2, batch_size, -1)
        hidden = torch.cat([hidden[:, 0], hidden[:, 1]], dim=2)
        cell = cell.view(self.encoder.num_layers, 2, batch_size, -1)
        cell = torch.cat([cell[:, 0], cell[:, 1]], dim=2)
        
        # Initialize decoder input
        decoder_input = torch.zeros(batch_size, 1, self.decoder.hidden_size).to(x.device)
        
        # Decode step by step
        outputs = []
        for t in range(x.size(1)):
            # Get current hidden state
            current_hidden = hidden[-1]  # Use last layer's hidden state
            
            # Calculate attention
            context, _ = self.attention(current_hidden, encoder_outputs)
            
            # Use context as decoder input
            decoder_input = context.unsqueeze(1)
            
            # Decoder step
            output, (hidden, cell) = self.decoder(decoder_input, (hidden, cell))
            
            # Project output
            step_output = self.output_layer(output.squeeze(1))
            outputs.append(step_output)
        
        # Combine outputs
        outputs = torch.stack(outputs, dim=1).squeeze(-1)
        
        return torch.sigmoid(outputs)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class VideoDataset(Dataset):
    def __init__(self, features_path, annotation_path):
        print("Initializing VideoDataset...")
        
        # Read features
        self.features = {}
        for file in os.listdir(features_path):
            if file.endswith('_features.npy'):
                # Remove '_features.npy' from filename
                video_id = file.replace('_features.npy', '')
                feature_path = os.path.join(features_path, file)
                self.features[video_id] = torch.from_numpy(np.load(feature_path))
        
        print(f"Found {len(self.features)} feature files")
        
        # Read annotations
        self.anno_df = pd.read_csv(annotation_path, sep='\t', header=None, 
                                 names=['video_id', 'category', 'scores'])
        
        # Get unique video IDs from annotations
        unique_videos = self.anno_df['video_id'].unique()
        print(f"Found {len(unique_videos)} unique videos in annotations")
        
        # Get intersection of videos that have both features and annotations
        self.video_ids = list(set(self.features.keys()).intersection(set(unique_videos)))
        print(f"Found {len(self.video_ids)} videos with both features and annotations")
        
        if len(self.video_ids) == 0:
            print("Features keys:", list(self.features.keys())[:5])
            print("Annotation video IDs:", list(unique_videos)[:5])
            raise ValueError("No matching videos found between features and annotations!")
        
        # Process annotations
        self.labels = {}
        for video_id in self.video_ids:
            # Get all annotations for this video
            video_annos = self.anno_df[self.anno_df['video_id'] == video_id]
            
            # Convert string scores to numpy arrays and average them
            all_scores = []
            for _, row in video_annos.iterrows():
                scores = np.array([float(x) for x in row['scores'].split(',')])
                # Normalize scores to 0-1 range
                scores = scores / 5.0  # scores are 1-5 in the dataset
                all_scores.append(scores)
            
            if len(all_scores) > 0:
                mean_scores = np.mean(all_scores, axis=0)
                self.labels[video_id] = torch.from_numpy(mean_scores).float()
            else:
                print(f"Warning: No annotations found for video {video_id}")
        
        # Find max sequence length for padding
        self.max_seq_len = max(features.shape[0] for features in self.features.values())
        print(f"Max sequence length: {self.max_seq_len}")

    def pad_sequence(self, sequence, max_len):
        """Pad sequence to max_len"""
        curr_len = sequence.shape[0]
        if curr_len >= max_len:
            return sequence[:max_len]
        else:
            padding = torch.zeros((max_len - curr_len, sequence.shape[1]), dtype=sequence.dtype)
            return torch.cat([sequence, padding], dim=0)

    def __len__(self):
        return len(self.video_ids)

    def __getitem__(self, idx):
        video_id = self.video_ids[idx]
        features = self.pad_sequence(self.features[video_id], self.max_seq_len)
        labels = self.pad_sequence(self.labels[video_id].unsqueeze(-1), self.max_seq_len).squeeze(-1)
        
        # Create attention mask (1 for real tokens, 0 for padding)
        mask = torch.zeros(self.max_seq_len, dtype=torch.bool)
        mask[:len(self.features[video_id])] = 1
        
        return features, labels, mask

class PSO:
    def __init__(self, n_particles, n_iterations, scores):
        self.scores = scores
        self.n_particles = n_particles
        self.n_iterations = n_iterations
        # Convert boolean positions to float
        self.positions = np.random.rand(n_particles, len(scores))
        self.velocities = np.random.randn(n_particles, len(scores)) * 0.1
        self.personal_best_pos = self.positions.copy()
        self.personal_best_score = np.array([self._evaluate(p > 0.5) for p in self.positions])
        self.global_best_pos = self.personal_best_pos[np.argmax(self.personal_best_score)]
        self.global_best_score = np.max(self.personal_best_score)

    def _evaluate(self, position):
        selected_scores = self.scores[position]
        return np.mean(selected_scores)

    def optimize(self):
        w, c1, c2 = 0.7, 1.5, 1.5
        
        for _ in range(self.n_iterations):
            r1, r2 = np.random.rand(), np.random.rand()
            
            # Update velocities using continuous positions
            self.velocities = (w * self.velocities + 
                             c1 * r1 * (self.personal_best_pos - self.positions) +
                             c2 * r2 * (self.global_best_pos - self.positions))
            
            # Update positions
            self.positions = self.positions + self.velocities
            
            # Clip positions to [0, 1]
            self.positions = np.clip(self.positions, 0, 1)
            
            # Evaluate using binary positions
            binary_positions = self.positions > 0.5
            scores = np.array([self._evaluate(p) for p in binary_positions])
            
            # Update personal best
            improved = scores > self.personal_best_score
            self.personal_best_pos[improved] = self.positions[improved]
            self.personal_best_score[improved] = scores[improved]
            
            # Update global best
            if np.max(scores) > self.global_best_score:
                self.global_best_pos = self.positions[np.argmax(scores)]
                self.global_best_score = np.max(scores)
        
        # Return final binary solution
        return self.global_best_pos > 0.5