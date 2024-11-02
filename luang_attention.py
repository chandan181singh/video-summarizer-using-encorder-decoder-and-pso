import torch
import torch.nn as nn
import torch.nn.functional as F

class LuongAttention(nn.Module):
    def __init__(self, hidden_size, method='dot'):
        super(LuongAttention, self).__init__()
        self.hidden_size = hidden_size
        self.method = method
        
        if method == 'general':
            self.W = nn.Linear(hidden_size, hidden_size, bias=False)
        elif method == 'concat':
            self.W = nn.Linear(hidden_size * 2, hidden_size, bias=False)
            self.v = nn.Parameter(torch.FloatTensor(hidden_size))

    def forward(self, decoder_hidden, encoder_outputs):
        # decoder_hidden: [batch_size, hidden_size]
        # encoder_outputs: [batch_size, seq_len, hidden_size]
        
        batch_size, seq_len, hidden_size = encoder_outputs.size()
        
        # Calculate attention scores
        if self.method == 'dot':
            # Simple dot product between decoder hidden state and encoder outputs
            attention_scores = torch.bmm(encoder_outputs, decoder_hidden.unsqueeze(2)).squeeze(2)
        
        elif self.method == 'general':
            # Linear transform on decoder hidden state before dot product
            energy = self.W(decoder_hidden)
            attention_scores = torch.bmm(encoder_outputs, energy.unsqueeze(2)).squeeze(2)
        
        elif self.method == 'concat':
            # Concatenate decoder hidden state with each encoder output and apply linear transform
            decoder_hidden_expanded = decoder_hidden.unsqueeze(1).expand(-1, seq_len, -1)
            concat = torch.cat((decoder_hidden_expanded, encoder_outputs), dim=2)
            energy = torch.tanh(self.W(concat))
            attention_scores = torch.sum(self.v * energy, dim=2)
        
        # Apply softmax to get attention weights
        attention_weights = F.softmax(attention_scores, dim=1)
        
        # Calculate context vector
        context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs).squeeze(1)
        
        return context, attention_weights

# Example usage:
if __name__ == "__main__":
    # Parameters
    batch_size = 32
    seq_len = 10
    hidden_size = 256
    
    # Create random test data
    decoder_hidden = torch.randn(batch_size, hidden_size)
    encoder_outputs = torch.randn(batch_size, seq_len, hidden_size)
    
    # Initialize attention mechanism
    attention = LuongAttention(hidden_size, method='general')
    
    # Get attention output
    context, attention_weights = attention(decoder_hidden, encoder_outputs)
    
    print(f"Context vector shape: {context.shape}")
    print(f"Attention weights shape: {attention_weights.shape}")
