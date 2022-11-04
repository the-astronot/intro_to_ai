import torchvision
from model import CNN_small
import numpy as np
import torch
from torch.utils.data import Subset
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt

# for this assignment, using a cpu should be sufficient if you don't have a gpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def plot_image(image):
    image = np.reshape(image, (3, 32, 32))
    image = np.transpose(image, (1, 2, 0))
    plt.imshow(image)
    plt.show()


def split_train_val(org_train_set, valid_ratio=0.1):

    num_train = len(org_train_set)

    split = int(np.floor(valid_ratio * num_train))

    indices = list(range(num_train))

    np.random.shuffle(indices)

    train_idx, val_idx = indices[split:], indices[:split]

    new_train_set = Subset(org_train_set, train_idx)
    val_set = Subset(org_train_set, val_idx)

    assert num_train - split == len(new_train_set)
    assert split == len(val_set)

    return new_train_set, val_set


def test(net, loader, device):
    # prepare model for testing (only important for dropout, batch norm, etc.)
    net.eval()

    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)

            output = net(data)
            test_loss += F.nll_loss(output, target, size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += (pred.eq(target.data.view_as(pred)).sum().item())

            total = total + 1

    print('Test set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
        test_loss, correct, len(loader.dataset),
        (100. * correct / len(loader.dataset))), flush=True)

    return 100.0 * correct / len(loader.dataset)


def train(net, loader, optimizer, epoch, device, log_interval=100):
    # prepare model for training (only important for dropout, batch norm, etc.)
    net.train()

    correct = 0
    for batch_idx, (data, target) in enumerate(loader):

        data, target = data.to(device), target.to(device)

        # clear up gradients for backprop
        optimizer.zero_grad()
        output = F.log_softmax(net(data), dim=1)

        # use NLL loss
        loss = F.nll_loss(output, target)

        # compute gradients and make updates
        loss.backward()
        optimizer.step()

        pred = output.data.max(1, keepdim=True)[1]
        correct += (pred.eq(target.data.view_as(pred)).sum().item())

        if batch_idx % log_interval == 0:
            print('\nTrain Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(loader.dataset), 100. * batch_idx / len(loader), loss.item()), flush=True)

    print('\tAccuracy: {:.2f}%'.format(100.0 * correct / len(loader.dataset)), flush=True)


if __name__ == '__main__':

    # set hyper-parameters
    train_batch_size = 100
    test_batch_size = 100
    n_epochs = 30
    learning_rate = 1e-2
    seed = 100
    input_dim = (3,32,32)
    out_dim = 10
    momentum = 0.9

    normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.225, 0.225, 0.225])
    train_transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), normalize])
    test_transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), normalize])

    train_dataset = torchvision.datasets.CIFAR10('./datasets/', train=True, download=True, transform=train_transforms)
    test_dataset = torchvision.datasets.CIFAR10('./datasets/', train=False, download=True, transform=test_transforms)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)

    # create neural network object
    network = CNN_small(in_dim=input_dim, out_dim=out_dim)
    network = network.to(device)

    # set up optimizer
    optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum)

    # training loop
    for epoch in range(1, n_epochs + 1):
        train(network, train_loader, optimizer, epoch, device)
        test(network, test_loader, device)
