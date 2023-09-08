import torch
import torch.nn as nn
import torch.nn.functional as F

class HousePredictorWithTransformerAttention(nn.Module):
    def __init__(self, input_dim=10, embed_dim=8, nhead=2):
        super(HousePredictorWithTransformerAttention, self).__init__()
        
        # Embedding layer for 'location'
        self.embedding = nn.Embedding(3, embed_dim)
        
        # MultiheadAttention layer
        self.attention = nn.MultiheadAttention(embed_dim=input_dim + embed_dim, num_heads=nhead)
        
        # Shared layers
        self.fc1 = nn.Linear(input_dim + embed_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        
        # Regression head for maintenance cost
        self.regression_head = nn.Linear(128, 1)
        
        # Classification head for house condition
        self.classification_head = nn.Linear(128, 5)
        
    def forward(self, x, location):
        embed = self.embedding(location)  # Convert location to dense vector
        x = torch.cat([x, embed], dim=1)  # Concatenate with other features
        
        # Reshape for multihead attention: (L, N, E) where L is target sequence length, N is batch size, E is embedding dim
        x = x.unsqueeze(0)  
        
        # Attention mechanism (note that we're using self-attention so Q=K=V=x)
        attn_output, _ = self.attention(x, x, x)
        x = attn_output.squeeze(0)
        
        # Shared layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        # Separate heads
        reg_output = self.regression_head(x)
        class_output = F.softmax(self.classification_head(x), dim=1)
        
        return reg_output, class_output

if __name__ == '__main__':
    model = HousePredictorWithTransformerAttention(input_dim=10)  # Excluding 'maintenance_cost', 'house_condition', and 'location' 
    print(model)
