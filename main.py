import os
import time

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

from models import efficientnet_modified
from trainer import train
from dataloader import Dataset300W, get_train_val , Transforms
from torch.optim import Adam


def main():

    num_epochs = 10
    output_size = 196
    batch_size = 16
    learning_rate = 0.00099
    n = 97
    model_path = "./saved/facial_keypoints.pth"

    if not (os.path.exists(model_path)):
        os.makedirs(model_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = Dataset300W(transform=Transforms())

    train_loader, valid_loader = get_train_val(dataset)

    model = efficientnet_modified()
    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)

    print("[INFOS]: starting the training loop....")
    loss_over_time = train(
        model, train_loader, valid_loader, optimizer, criterion, num_epochs, device
    )
    
    torch.save(model.state_dict(), model_path)


if __name__ == "__main__":
    # argparse

    main()
