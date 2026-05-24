import torch
import torch.nn as nn

class BisindoMLP(nn.Module):
    def __init__(self,dropout_rate=0.3):
        super().__init__()
        
        self.layer = nn.Sequential(
            nn.Linear(126,256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256,128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128,26),
        )
    def forward(self,x):
        return self.layer(x)
    
    
if __name__ == "__main__":
    model = BisindoMLP()
    print(model)
    
    dummy_input = torch.randn(32, 126)
    output = model(dummy_input)
    
    print(f"Input shape:  {dummy_input.shape}")
    print(f"Output shape: {output.shape}")