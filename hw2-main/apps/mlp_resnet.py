import sys
from needle.data import MNISTDataset, DataLoader
sys.path.append("../python")
import needle as ndl
import needle.nn as nn
import needle.data as data
import numpy as np
import time
import os
import pdb
from tqdm import tqdm

np.random.seed(0)
# MY_DEVICE = ndl.backend_selection.cuda()


def ResidualBlock(dim, hidden_dim, norm=nn.BatchNorm1d, drop_prob=0.1):
    ### BEGIN YOUR SOLUTION
    # main block. Recall that Sequential applies operations onto X sequentially
    # which is what we want (as seen in the image)
    block = nn.Sequential(nn.Linear(in_features=dim, out_features=hidden_dim),
                  norm(hidden_dim),
                  nn.ReLU(),
                  nn.Dropout(drop_prob),
                  nn.Linear(in_features=hidden_dim, out_features=dim),
                  norm(dim)
                  )
    # create residual link
    res = nn.Residual(block)
    # then, we need to pass X through the residual block
    # and then finally through Relu ==> we need a sequential block
    return nn.Sequential(res, nn.ReLU())


    ### END YOUR SOLUTION


def MLPResNet(
    dim,
    hidden_dim=100,
    num_blocks=3,
    num_classes=10,
    norm=nn.BatchNorm1d,
    drop_prob=0.1,
):
    ### BEGIN YOUR SOLUTION
    # again, sequentialy apply the layers to the input.
    # can't do ResidualBlock * num to get num residual blocks
    blocks = []
    blocks.append(nn.Linear(in_features=dim, out_features=hidden_dim))
    blocks.append(nn.ReLU())
    for __ in range(num_blocks):
        blocks.append(ResidualBlock(dim=hidden_dim, hidden_dim=hidden_dim//2, norm=norm, drop_prob=drop_prob))
    blocks.append(nn.Linear(in_features=hidden_dim, out_features=num_classes))
    # need to sequentially apply blocks to input
    return nn.Sequential(*blocks)

def epoch(dataloader, model, opt=None):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    tot_loss, tot_error = [], 0.0
    loss_fn = nn.SoftmaxLoss()
    if opt is None:    
        model.eval()
        for X, y in dataloader:
            logits = model(X)
            loss = loss_fn(logits, y)
            # accuracy is predicted - expected
            tot_error += np.sum(logits.numpy().argmax(axis=1) != y.numpy())
            # SoftmaxLoss function returns a tensor that is not always 1x1
            # so, we need to append the losses to a list and then later 
            # take the mean.
            tot_loss.append(loss.numpy())
    else:
        model.train()
        for X, y in tqdm(dataloader, position=0, leave=True):
            opt.reset_grad()
            logits = model(X)
            loss = loss_fn(logits, y)
            tot_error += np.sum(logits.numpy().argmax(axis=1) != y.numpy())
            tot_loss.append(loss.numpy())
            # we need to reset the grad because accumulation of past grads
            # and current grads will mess up the calculuation of the gradients
            # and result in an incorrect update to the params.
            # calculate the grads
            loss.backward()
            # use grads to perform update
            opt.step()
            # pdb.set_trace()
    sample_nums = len(dataloader.dataset)
    return tot_error/sample_nums, np.mean(tot_loss)
    ### END YOUR SOLUTION


def train_mnist(
    batch_size=100,
    epochs=10,
    optimizer=ndl.optim.Adam,
    lr=0.001,
    weight_decay=0.001,
    hidden_dim=100,
    data_dir="data",
):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    # this is how they did it in the test code.
    if optimizer is not None:
        train_set = MNISTDataset(f"{data_dir}/train-images-idx3-ubyte.gz", 
                                f"{data_dir}/train-labels-idx1-ubyte.gz")
        test_set = MNISTDataset(f"{data_dir}/t10k-images-idx3-ubyte.gz",
                                f"{data_dir}/t10k-labels-idx1-ubyte.gz")
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
        #input must be 784 because an MNIST imgage is 28x28x1.
        model = MLPResNet(dim=784, hidden_dim=hidden_dim)
        # pdb.set_trace()
        opt = optimizer(model.parameters(), lr=lr, weight_decay=weight_decay)
        # print("| Epoch | Train Accuracy | Train Loss | Test Accuracy| Test Loss |")
        for ep in range(epochs):
            train_acc, train_loss = epoch(train_loader, model, opt=opt)
            print("| ", ep, " |", train_acc, " |", train_loss)
        print("test acc, and test loss")
        test_acc, test_loss = epoch(test_loader, model, opt=None)
        print("|", epochs, " |","| ", test_acc, " |", test_loss, " |")
        # evaluating the model ==> opt = None
        return np.array([train_acc, train_loss, test_acc, test_loss])
    ### END YOUR SOLUTION


if __name__ == "__main__":
    train_mnist(data_dir="../data")
