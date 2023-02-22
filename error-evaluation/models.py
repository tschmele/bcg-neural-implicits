import torch
import torch as th


# --- Model ---
class SimpleModel(th.nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()

        # first hidden layer
        self.fc1 = th.nn.Linear(3, 32, dtype=torch.float64)
        self.act1 = th.nn.ReLU()
        # second hidden layer
        self.fc2 = th.nn.Linear(32, 32, dtype=torch.float64)
        self.act2 = th.nn.ReLU()
        # third hidden layer
        self.fc3 = th.nn.Linear(32, 32, dtype=torch.float64)
        self.act3 = th.nn.ReLU()
        # fourth hidden layer
        self.fc4 = th.nn.Linear(32, 32, dtype=torch.float64)
        self.act4 = th.nn.ReLU()
        # fifth hidden layer
        self.fc5 = th.nn.Linear(32, 32, dtype=torch.float64)
        self.act5 = th.nn.ReLU()
        # sixth hidden layer
        self.fc6 = th.nn.Linear(32, 32, dtype=torch.float64)
        self.act6 = th.nn.ReLU()
        # seventh hidden layer
        self.fc7 = th.nn.Linear(32, 32, dtype=torch.float64)
        self.act7 = th.nn.ReLU()
        # eighth hidden layer
        self.fc8 = th.nn.Linear(32, 32, dtype=torch.float64)
        self.act8 = th.nn.ReLU()
        # output layer
        self.fc9 = th.nn.Linear(32, 1, dtype=torch.float64)

    # The forward function defines the computation in the forward direction. It is called by pytorch and when
    # evaluating the model. The input is simply passed through the layers as defined in the constructor.
    def forward(self, x):
        # Run the input through all the layers
        x = self.fc1(x)
        x = self.act1(x)
        x = self.fc2(x)
        x = self.act2(x)
        x = self.fc3(x)
        x = self.act3(x)
        x = self.fc4(x)
        x = self.act4(x)
        x = self.fc5(x)
        x = self.act5(x)
        x = self.fc6(x)
        x = self.act6(x)
        x = self.fc7(x)
        x = self.act7(x)
        x = self.fc8(x)
        x = self.act8(x)
        x = self.fc9(x)
        x = x.squeeze(-1)
        # Return the output
        return x
