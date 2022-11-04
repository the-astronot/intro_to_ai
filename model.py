import torch
import torch.nn as nn
import torch.nn.functional as F

def calc_out_dim(in_dim, padding, dilation, kernel_size, stride) -> int:
    return (in_dim + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1

class FC(nn.Module):

    def __init__(self, in_dim, out_dim, num_hidden_layers, layer_size):
        super().__init__()

        self.num_layers = num_hidden_layers * 2 + 3 # *2 accounts for ReLU layers, +3 is input layer, input relu layer, output layer

        self.in_dim = in_dim
        self.out_dim = out_dim

        self.layer_size = layer_size

        self.layer_list = nn.ModuleList()

        self.layer_list.append(nn.Linear(self.in_dim, self.layer_size))
        self.num_hidden_layers = num_hidden_layers

        for i in range(1,self.num_hidden_layers):
            self.layer_list.append(nn.Linear(self.layer_size, self.layer_size))


        self.layer_list.append(nn.Linear(self.layer_size, self.out_dim))

    def forward(self, x):

        x = x.view(-1, self.in_dim)

        for i in range(self.num_hidden_layers):
            x = F.relu(self.layer_list[i](x))

        return self.layer_list[self.num_hidden_layers](x)

class CNN(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim

        self.fc_layer1_neurons = 500
        self.fc_layer2_neurons = 500
        self.fc_layer3_neurons = 500

        self.conv1_filters = 96
        self.conv2_filters = 256
        self.conv3_filters = 384
        self.conv4_filters = 384
        self.conv5_filters = 256

        self.conv1_kernel_size = (11,11)
        self.conv2_kernel_size = (5,5)
        self.conv3_kernel_size = (3,3)
        self.conv4_kernel_size = (3,3)
        self.conv5_kernel_size = (3,3)

        self.maxpool1_kernel_size = (3,3)
        self.maxpool2_kernel_size = (3,3)
        self.maxpool3_kernel_size = (3,3)

        self.conv1_stride = 4
        self.conv2_stride = 1
        self.conv3_stride = 1
        self.conv4_stride = 1
        self.conv5_stride = 1

        self.maxpool1_stride = 2
        self.maxpool2_stride = 2
        self.maxpool3_stride = 2

        self.conv1_padding = 0
        self.conv2_padding = 2
        self.conv3_padding = 1
        self.conv4_padding = 1
        self.conv5_padding = 1

        self.conv1_dim_h = calc_out_dim(self.in_dim[1], self.conv1_padding, 1, self.conv1_kernel_size[0], self.conv1_stride)
        self.conv1_dim_w = calc_out_dim(self.in_dim[2], self.conv1_padding, 1, self.conv1_kernel_size[1], self.conv1_stride)

        self.maxpool1_dim_h = calc_out_dim(self.conv1_dim_h, 0, 1, self.maxpool1_kernel_size[0], self.maxpool1_stride)
        self.maxpool1_dim_w = calc_out_dim(self.conv1_dim_w, 0, 1, self.maxpool1_kernel_size[1], self.maxpool1_stride)

        self.conv2_dim_h = calc_out_dim(self.maxpool1_dim_h, self.conv2_padding, 1, self.conv2_kernel_size[0], self.conv2_stride)
        self.conv2_dim_w = calc_out_dim(self.maxpool1_dim_w, self.conv2_padding, 1, self.conv2_kernel_size[1], self.conv2_stride)

        self.maxpool2_dim_h = calc_out_dim(self.conv2_dim_h, 0, 1, self.maxpool2_kernel_size[0], self.maxpool2_stride)
        self.maxpool2_dim_w = calc_out_dim(self.conv2_dim_w, 0, 1, self.maxpool2_kernel_size[1], self.maxpool2_stride)

        self.conv3_dim_h = calc_out_dim(self.maxpool2_dim_h, self.conv3_padding, 1, self.conv3_kernel_size[0], self.conv3_stride)
        self.conv3_dim_w = calc_out_dim(self.maxpool2_dim_w, self.conv3_padding, 1, self.conv3_kernel_size[1], self.conv3_stride)

        self.conv4_dim_h = calc_out_dim(self.conv3_dim_h, self.conv4_padding, 1, self.conv4_kernel_size[0], self.conv4_stride)
        self.conv4_dim_w = calc_out_dim(self.conv3_dim_w, self.conv4_padding, 1, self.conv4_kernel_size[1], self.conv4_stride)

        self.conv5_dim_h = calc_out_dim(self.conv4_dim_h, self.conv5_padding, 1, self.conv5_kernel_size[0], self.conv5_stride)
        self.conv5_dim_w = calc_out_dim(self.conv4_dim_w, self.conv5_padding, 1, self.conv5_kernel_size[1], self.conv5_stride)

        self.maxpool3_dim_h = calc_out_dim(self.conv5_dim_h, 0, 1, self.maxpool3_kernel_size[0], self.maxpool3_stride)
        self.maxpool3_dim_w = calc_out_dim(self.conv5_dim_w, 0, 1, self.maxpool3_kernel_size[1], self.maxpool3_stride)

        self.conv1 = nn.Conv2d(3, self.conv1_filters, self.conv1_kernel_size, stride=self.conv1_stride, padding=self.conv1_padding)
        self.conv2 = nn.Conv2d(self.conv1_filters, self.conv2_filters, self.conv2_kernel_size, stride=self.conv2_stride, padding=self.conv2_padding)
        self.conv3 = nn.Conv2d(self.conv2_filters, self.conv3_filters, self.conv3_kernel_size, stride=self.conv3_stride, padding=self.conv3_padding)
        self.conv4 = nn.Conv2d(self.conv3_filters, self.conv4_filters, self.conv4_kernel_size, stride=self.conv4_stride, padding=self.conv4_padding)
        self.conv5 = nn.Conv2d(self.conv4_filters, self.conv5_filters, self.conv5_kernel_size, stride=self.conv5_stride, padding=self.conv5_padding)

        self.maxpool1 = nn.MaxPool2d(self.maxpool1_kernel_size, self.maxpool1_stride)
        self.maxpool2 = nn.MaxPool2d(self.maxpool2_kernel_size, self.maxpool2_stride)
        self.maxpool3 = nn.MaxPool2d(self.maxpool3_kernel_size, self.maxpool3_stride)


        self.dropout1 = nn.Dropout2d(0.5)
        self.dropout2 = nn.Dropout(0.5)

        self.fc_inputs = int(self.conv5_filters * self.maxpool3_dim_w * self.maxpool3_dim_h)

        self.lin1 = nn.Linear(self.fc_inputs, self.fc_layer1_neurons)
        self.lin2 = nn.Linear(self.fc_layer1_neurons, self.fc_layer2_neurons)
        self.lin3 = nn.Linear(self.fc_layer2_neurons, self.fc_layer3_neurons)
        self.lin4 = nn.Linear(self.fc_layer3_neurons, self.out_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.maxpool1(x)
        x = F.relu(self.conv2(x))
        x = self.maxpool2(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = self.maxpool3(x)
        x = self.dropout1(x)

        # flatten convolutional layer into vector
        x = x.view(x.size(0), -1)
        x = F.relu(self.lin1(x))
        x = self.dropout2(x)
        x = F.relu(self.lin2(x))
        x = F.relu(self.lin3(x))
        x = self.lin4(x)
        return x

# 3*96*11*11+96 + 96*256*5*5+256 + 256*384*3*3+384 + 384*384*3*3+384 + 384*256*3*3+256 + 3840*500+500 + 500*500+500 + 500*500+500 + 500*11+11 = 6_174_211 parameters
# Train Epoch: 16 [10062/12978 (99%)]     Loss: 0.008765
#         Accuracy: 99.38%
# Test set: Avg. loss: -15285.4297, Accuracy: 524/539 (97.22%)

class CNN_small(nn.Module):

    def __init__(self, in_dim, out_dim):
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim

        self.fc_layer1_neurons = 400
        self.fc_layer2_neurons = 200

        self.layer1_filters = 32
        self.layer2_filters = 128

        self.layer1_kernel_size = (4,4)
        self.layer2_kernel_size = (4,4)
        self.layer1_stride = 1
        self.layer2_stride = 1
        self.layer1_padding = 0
        self.layer2_padding = 0

        #NB: these calculations assume:
        #1) padding is 0;
        #2) stride is picked such that the last step ends on the last pixel, i.e., padding is not used
        self.layer1_dim_h = (self.in_dim[1] - self.layer1_kernel_size[0]) / self.layer1_stride + 1
        self.layer1_dim_w = (self.in_dim[2] - self.layer1_kernel_size[1]) / self.layer1_stride + 1

        self.layer2_dim_h = (self.layer1_dim_h - self.layer2_kernel_size[0]) / self.layer2_stride + 1
        self.layer2_dim_w = (self.layer1_dim_w - self.layer2_kernel_size[1]) / self.layer2_stride + 1

        self.conv1 = nn.Conv2d(3, self.layer1_filters, self.layer1_kernel_size, stride=self.layer1_stride, padding=self.layer1_padding)

        self.conv2 = nn.Conv2d(self.layer1_filters, self.layer2_filters, self.layer2_kernel_size, stride=self.layer2_stride, padding=self.layer2_padding)

        self.fc_inputs = int(self.layer2_filters * self.layer2_dim_h * self.layer2_dim_w)

        self.lin1 = nn.Linear(self.fc_inputs, self.fc_layer1_neurons)

        self.lin2 = nn.Linear(self.fc_layer1_neurons, self.fc_layer2_neurons)

        self.lin3 = nn.Linear(self.fc_layer2_neurons, self.out_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        # flatten convolutional layer into vector
        x = x.view(x.size(0), -1)
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        x = self.lin3(x)
        return x
