import torch
import torch.nn as nn

class AdaptiveLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Gates with clarity score input
        self.forget_gate = nn.Linear(input_size + hidden_size + 1, hidden_size)
        self.input_gate = nn.Linear(input_size + hidden_size + 1, hidden_size)
        self.output_gate = nn.Linear(input_size + hidden_size + 1, hidden_size)
        self.cell_gate = nn.Linear(input_size + hidden_size, hidden_size)
        
    def forward(self, x, h_prev, c_prev, clarity_score):
        # Combine input and previous hidden state
        combined = torch.cat([x, h_prev], dim=1)
        combined_with_clarity = torch.cat([combined, clarity_score.unsqueeze(1)], dim=1)
        
        # Compute gates
        f_t = torch.sigmoid(self.forget_gate(combined_with_clarity))
        i_t = torch.sigmoid(self.input_gate(combined_with_clarity))
        o_t = torch.sigmoid(self.output_gate(combined_with_clarity))
        
        # Update cell state
        c_tilde = torch.tanh(self.cell_gate(combined))
        c_t = f_t * c_prev + i_t * c_tilde
        
        # Compute hidden state
        h_t = o_t * torch.tanh(c_t)
        
        return h_t, c_t

class AdaptiveLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.hidden_size = hidden_size
        
        # Adaptive LSTM cell
        self.lstm_cell = AdaptiveLSTMCell(input_size, hidden_size)
        
        # Classification layer
        self.classifier = nn.Linear(hidden_size, num_classes)
        
        # Embedding similarity layer
        self.similarity = nn.CosineSimilarity(dim=1)
        
    def forward(self, x, clarity_scores):
        batch_size = x.size(0)
        seq_length = x.size(1)
        
        # Initialize hidden state and cell state
        h_t = torch.zeros(batch_size, self.hidden_size).to(x.device)
        c_t = torch.zeros(batch_size, self.hidden_size).to(x.device)
        
        # Process sequence
        for t in range(seq_length):
            h_t, c_t = self.lstm_cell(
                x[:, t, :],  # Current input
                h_t,         # Previous hidden state
                c_t,         # Previous cell state
                clarity_scores[:, t]  # Current clarity score
            )
        
        # Final classification
        output = self.classifier(h_t)
        return output
