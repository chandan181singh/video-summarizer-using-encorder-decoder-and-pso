import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset
import os
import pandas as pd
import math
from luong_attention import LuongAttention
from bahdanau_attention import BahdanauAttention
from path import ATTENTION_MODEL

class VideoSummarizer(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=256, num_layers=2, dropout=0.2):
        super(VideoSummarizer, self).__init__()
        
        # Input projection without batch normalization first
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Layer normalization instead of batch normalization
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
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
        
        # Attention mechanism (Luong)
        if ATTENTION_MODEL == "Luong":
            self.attention = LuongAttention(hidden_dim * 2, method='general')  # Double size to match encoder output
        else:
            self.attention = BahdanauAttention(hidden_dim * 2, hidden_dim * 2)
        
        # Update output layer with additional processing
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # Project input and apply layer normalization
        x = self.input_proj(x)
        x = self.layer_norm(x)
        
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
        from path import DATASET, DATASET_CONFIG
        
        self.dataset_type = DATASET
        self.config = DATASET_CONFIG[DATASET]
        
        # Read features
        self.features = {}
        for file in os.listdir(features_path):
            if file.endswith('_features.npy'):
                video_id = file.replace('_features.npy', '')
                feature_path = os.path.join(features_path, file)
                self.features[video_id] = torch.from_numpy(np.load(feature_path))
        
        print(f"Found {len(self.features)} feature files")
        
        # Read annotations based on dataset type
        if self.dataset_type == "SumMe":
            self.load_summe_annotations(annotation_path)
        else:  # TVSum
            self.load_tvsum_annotations(annotation_path)
        
        # Find max sequence length for padding
        self.max_seq_len = max(features.shape[0] for features in self.features.values())
        print(f"Max sequence length: {self.max_seq_len}")

    def load_summe_annotations(self, annotation_path):
        import scipy.io as sio
        self.labels = {}
        self.categories = {}
        
        for video_id in self.features.keys():
            mat_file = os.path.join(annotation_path, f"{video_id}.mat")
            if os.path.exists(mat_file):
                mat_data = sio.loadmat(mat_file)
                user_scores = mat_data['user_score']
                
                # Normalize scores to increase positive labels
                normalized_scores = []
                for user_idx in range(user_scores.shape[1]):
                    scores = user_scores[:, user_idx]
                    # More aggressive normalization to increase positive labels
                    scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-10)
                    scores = scores ** 0.7  # Power < 1 increases values
                    normalized_scores.append(scores)
                
                mean_scores = np.mean(normalized_scores, axis=0)
                self.labels[video_id] = torch.from_numpy(mean_scores).float()
                self.categories[video_id] = 'NA'
        
        # Create dummy DataFrame for compatibility
        self.anno_df = pd.DataFrame({
            'video_id': list(self.labels.keys()),
            'category': ['NA'] * len(self.labels),
            'scores': [self.labels[vid].numpy().tolist() for vid in self.labels.keys()]
        })
        
        self.video_ids = list(self.labels.keys())

    def load_tvsum_annotations(self, annotation_path):
        self.anno_df = pd.read_csv(annotation_path, sep='\t', header=None,
                                 names=['video_id', 'category', 'scores'])
        
        unique_videos = self.anno_df['video_id'].unique()
        self.video_ids = list(set(self.features.keys()).intersection(set(unique_videos)))
        
        # Process annotations
        self.labels = {}
        for video_id in self.video_ids:
            video_annos = self.anno_df[self.anno_df['video_id'] == video_id]
            all_scores = []
            for _, row in video_annos.iterrows():
                scores = np.array([float(x) for x in row['scores'].split(',')])
                scores = scores / self.config['score_range'][1]  # Normalize to [0,1]
                all_scores.append(scores)
            
            if len(all_scores) > 0:
                mean_scores = np.mean(all_scores, axis=0)
                self.labels[video_id] = torch.from_numpy(mean_scores).float()

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
        from path import DATASET, DATASET_CONFIG
        
        self.scores = scores
        self.n_particles = n_particles
        self.n_iterations = n_iterations
        self.max_summary_length = DATASET_CONFIG[DATASET]['max_summary_length'] * 1.5
        
        # Initialize positions with higher probability of selection
        self.positions = np.random.rand(n_particles, len(scores)) * 1.2
        self._adjust_positions_to_length_constraint()
        
        self.velocities = np.random.randn(n_particles, len(scores)) * 0.1
        self.personal_best_pos = self.positions.copy()
        self.personal_best_score = np.array([self._evaluate(p > 0.5) for p in self.positions])
        self.global_best_pos = self.personal_best_pos[np.argmax(self.personal_best_score)]
        self.global_best_score = np.max(self.personal_best_score)

    def _adjust_positions_to_length_constraint(self):
        """Adjust positions to satisfy the summary length constraint"""
        for i in range(len(self.positions)):
            # Sort positions
            sorted_indices = np.argsort(self.positions[i])
            # Set top k% to 1, rest to 0
            k = int(len(self.positions[i]) * self.max_summary_length)
            self.positions[i][sorted_indices[:-k]] = 0
            self.positions[i][sorted_indices[-k:]] = 1

    def _evaluate(self, position):
        selected_scores = self.scores[position]
        # Reward selecting more frames
        selection_bonus = position.sum() / len(position) * 0.2
        return np.mean(selected_scores) + selection_bonus

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