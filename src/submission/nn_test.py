import torch
import torch.nn as nn

class FeedForwardNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(FeedForwardNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.hidden_layer = nn.Linear(embedding_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        embedded = self.embedding(x)
        hidden_output = torch.relu(self.hidden_layer(embedded))
        output = self.output_layer(hidden_output)
        return output

# Example usage
# Assume vocab_size = 10000, embedding_dim = 100, hidden_dim = 50, output_dim = 10
vocab_size = 10000
embedding_dim = 100
hidden_dim = 50
output_dim = 10

# Create the model
model = FeedForwardNN(vocab_size, embedding_dim, hidden_dim, output_dim)

# Input example: a tensor of size (batch_size, sequence_length)
# Replace this with your actual input data
input_example = torch.randint(0, vocab_size, (32, 20))  # Batch size of 32, sequence length of 20

# Forward pass
output = model(input_example)

# Print the output shape
print("Output shape:", output.shape)

