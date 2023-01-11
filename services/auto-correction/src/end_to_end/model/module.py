import torch.nn as nn

class LayerClassifier(nn.Module):
    def __init__(self, input_dim, output_dim, drop_rate=0.0) -> None:
        super().__init__()
        
        self.dropout = nn.Dropout(drop_rate)
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = self.dropout(x)
        return self.linear(x)