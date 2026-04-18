import torch


class MultiLayerPerceptron(torch.nn.Module):
    def __init__(
        self, feat_in, feat_out, hidden_dim=None, activation="ReLU", stack_size=2
    ):
        super().__init__()

        layers = []
        prev_dim = feat_in

        if hidden_dim is None:
            hidden_dim = feat_out

        activation = getattr(torch.nn, activation)

        for _ in range(stack_size - 1):
            layers.append(torch.nn.Linear(prev_dim, hidden_dim))
            layers.append(activation())
            prev_dim = hidden_dim

        layers.append(torch.nn.Linear(prev_dim, feat_out))
        self.network = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
