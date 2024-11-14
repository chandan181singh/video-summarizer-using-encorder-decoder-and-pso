import torch
import torch.nn as nn
import torch.nn.functional as F

class BahdanauAttention(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, attention_dim=256):
        super(BahdanauAttention, self).__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)  # Transform encoder hidden state
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)  # Transform decoder hidden state
        self.attention = nn.Linear(attention_dim, 1)  # Compute attention scores

    def forward(self, decoder_hidden, encoder_outputs):
        # Step 1: Transform encoder outputs
        encoder_scores = self.encoder_att(encoder_outputs)  # (batch_size, seq_len, attention_dim)
        
        # Step 2: Transform decoder hidden state and align dimensions
        decoder_scores = self.decoder_att(decoder_hidden).unsqueeze(1)  # (batch_size, 1, attention_dim)
        
        # Step 3: Compute attention scores using additive mechanism
        attention_scores = torch.tanh(encoder_scores + decoder_scores)
        attention_scores = self.attention(attention_scores).squeeze(-1)  # (batch_size, seq_len)
        
        # Step 4: Compute attention weights using softmax
        attention_weights = F.softmax(attention_scores, dim=1)  # (batch_size, seq_len)
        
        # Step 5: Compute context vector as the weighted sum of encoder outputs
        context_vector = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs).squeeze(1)  # (batch_size, encoder_dim)

        return context_vector, attention_weights
