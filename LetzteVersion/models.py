import torch
import torch as th
import util

# --- Model ---
# Defines simple model by subclassing th.nn.Module
class SimpleModel(th.nn.Module):
    # Constructor. Takes the input shape (channels (nist), height, width) and the number of classes
    def __init__(self):
        super(SimpleModel, self).__init__()

        # flatten input image
        #self.flatten = th.nn.Flatten()
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
        #x = self.flatten(x)
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

# --- Model ---
# Better version of the model with some additional convolutional layers before the fully connected ones.
class CNN(th.nn.Module):
    # Constructor. Takes the input shape (channels, height, width) and the number of classes
    def __init__(self, input_shape, classes):
        super(CNN, self).__init__()

        # Define the convolutional layers
        self.conv1 = th.nn.Conv2d(input_shape[2], 16, kernel_size=3)
        self.relu1 = th.nn.ELU()
        self.pool1 = th.nn.MaxPool2d(kernel_size=2, stride=2) # 2x2 max pooling

        self.conv2 = th.nn.Conv2d(16, 32, kernel_size=3)
        self.relu2 = th.nn.ELU()

        # Calculate size of the dense layer. This can be done manually but is quite annoying.
        # The helper function is defined in util.py
        # A neat side effect is, that we can use the CNN class for any input size we want.
        conv_out_dim = util.conv2d_out_dim(input_shape[0:2], (3, 3), (1, 1), (1, 1), (0, 0)) # conv1
        conv_out_dim = (conv_out_dim[0] // 2, conv_out_dim[1] // 2)                           # pool1
        conv_out_dim = util.conv2d_out_dim(conv_out_dim,    (3, 3), (1, 1), (1, 1), (0, 0)) # conv2

        # Define the dense layers
        self.fc1 = th.nn.Linear(conv_out_dim[0] * conv_out_dim[1] * self.conv2.out_channels, 16)
        self.relu3 = th.nn.ELU()
        self.fc2 = th.nn.Linear(16, 16)
        self.relu4 = th.nn.ELU()
        self.fc3 = th.nn.Linear(16, classes)

    # The forward function defines the computation in the forward direction. It is called by pytorch and when
    # evaluating the model. The input is simply passed through the layers as defined in the constructor.
    def forward(self, x):
        # Run the input through all the layers
        # Convolutional block 1
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        # Convolutional block 2
        x = self.conv2(x)
        x = self.relu2(x)
        # Dense block
        x = x.view(x.size(0), -1)   # flatten output into large vector
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        x = self.relu4(x)
        x = self.fc3(x)
        # Return the output
        return x
