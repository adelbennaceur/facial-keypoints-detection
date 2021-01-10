import torch
import torch.nn as nn
import torchvision
from torchvision import transforms, utils, models
import numpy as np


def train(model, train_loader, val_loader, optimizer, criterion, num_epochs, device):
    loss_o_t = []
    for epoch in range(num_epochs):
        running_loss = 0.0
        train_loss = 0.0
        model.train()
        for step in range(1, len(train_loader) + 1):

            batch = next(iter(train_loader))

            images, labels = batch["image"], batch["landmarks"]
            images, labels = images.to(device), labels.to(device)
            labels = labels.reshape(labels.size(0), -1)

            outs = model(images)
            loss = criterion(outs, labels)

            optimizer.zero_grad()
            loss.backward()

            train_loss += loss.item()
            running_loss = train_loss / (step)

            optimizer.step()

            print(
                "Epoch [{}/{}] Training loss : {:.4f}".format(
                    epoch + 1, num_epochs, running_loss
                )
            )

        model.eval()
        with torch.no_grad():
            val_loss, running_loss = validate(
                model, val_loader, criterion, running_loss
            )
            print(
                "Epoch [{}/{}] Validation loss : {:.4f}".format(
                    epoch + 1, num_epochs, running_loss
                )
            )

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        loss_o_t.append(train_loss)
        print("\n--------------------------------------------------")
        print(
            "Epoch: {}  Train Loss: {:.4f}  Valid Loss: {:.4f}".format(
                epoch, train_loss, val_loss
            )
        )
        print("--------------------------------------------------")
    print("Finished Training")
    return loss_o_t


def validate(model, val_loader, criterion, running_loss, device):
    """
    function to validate the model and change the high level parameters during the training
    """
    # feedfroward during the the training and calculating the loss and appending to the loss
    val_loss = 0
    val_accuracy = 0

    for step in range(1, len(val_loader) + 1):

        batch = next(iter(val_loader))
        # unpacking the images and the correspening labelimages , labels = data
        images, labels = batch["image"], batch["landmarks"]
        images, labels = images.to(device), labels.to(device)

        # reshaping the labels
        labels = labels.reshape(labels.shape[0], -1)
        output = model(images)
        # computing the loss
        loss = criterion(output, labels)
        val_loss += loss.item()
        running_loss = val_loss / step

    return val_loss, running_loss
