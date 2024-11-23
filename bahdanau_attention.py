import torch
import torch.nn as nn

class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size, method='concat'):
        super(BahdanauAttention, self).__init__()
        self.hidden_size = hidden_size
        self.method = method
        
        # Attention layers
        self.attention_hidden = nn.Linear(hidden_size, hidden_size)
        self.attention_context = nn.Linear(hidden_size, hidden_size)
        self.v = nn.Parameter(torch.randn(hidden_size))
        
    def forward(self, hidden, encoder_outputs):
      
        seq_len = encoder_outputs.size(1)
        
        # Reshape hidden to (batch_size, 1, hidden_size) for broadcasting
        hidden = hidden.unsqueeze(1)
        
        # Calculate attention scores
        hidden_proj = self.attention_hidden(hidden)  # (batch_size, 1, hidden_size)
        context_proj = self.attention_context(encoder_outputs)  # (batch_size, seq_len, hidden_size)
        
        # Combine projections and apply tanh
        energy = torch.tanh(hidden_proj + context_proj)  # (batch_size, seq_len, hidden_size)
        
        # Calculate attention weights
        v = self.v.repeat(encoder_outputs.size(0), 1).unsqueeze(1)  # (batch_size, 1, hidden_size)
        attention_weights = torch.bmm(energy, v.transpose(1, 2)).squeeze(2)  # (batch_size, seq_len)
        attention_weights = torch.softmax(attention_weights, dim=1)
        
        # Calculate context vector
        context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs).squeeze(1)
        
        return context, attention_weights
