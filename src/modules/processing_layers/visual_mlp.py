import torch.nn as nn

class VideoFeatureMLP(nn.Module):
    """Converts a video feature tensor into a single feature vector using a multi-layer perceptron.

    Args:
        input_features: hidden feature dim
        output_features: output feature dim
        hidden_layers: list of hidden layer sizes
        dropout: dropout probability
    """
    def __init__(self, input_features, output_features, hidden_layers, dropout):
        super(VideoFeatureMLP, self).__init__()
        layers = []
        in_features = input_features

        # Create hidden layers
        for hidden_features in hidden_layers:
            layers.append(nn.Linear(in_features, hidden_features))
            layers.append(nn.ReLU(inplace=True))
            # layers.append(nn.BatchNorm1d(hidden_features))
            layers.append(nn.Dropout(p=dropout))
            in_features = hidden_features
        
        # Output layer
        layers.append(nn.Linear(in_features, output_features))
        
        # Register all layers
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.layers(x)
