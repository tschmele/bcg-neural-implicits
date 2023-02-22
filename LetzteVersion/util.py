import torch as th
from torchvision import datasets, transforms

# Computes the output dimensions of a convolutional layer given input dimensions and layer parameters
# See https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
# Returns a tuple (output_height, output_width)
def conv2d_out_dim(in_dim, kernel_size, stride, dilation, padding):
    return (
        int((in_dim[0] + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) / stride[0] + 1),     # height
        int((in_dim[1] + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) / stride[1] + 1)      # width
    )