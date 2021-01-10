import time
import os

import torch
import matplotlib.pyplot as plt
import numpy as np


from models import efficientnet_modified
from dataloader import Transforms, Dataset300W, get_train_val


def evaluate(model_path, valid_loader, device):

    start_time = time.time()
    with torch.no_grad():

        network = efficientnet_modified()
        network.to(device)
        network.load_state_dict(torch.load(model_path))

        network.eval()

        images, landmarks = next(iter(valid_loader))

        images = images.to(device)
        landmarks = (landmarks + 0.5) * 224

        preds = (network(images).cpu() + 0.5) * 224
        preds = preds.view(-1, 68, 2)

        plt.figure(figsize=(10, 40))

        for img_num in range(4):
            plt.subplot(4, 1, img_num + 1)
            plt.imshow(
                images[img_num].cpu().numpy().transpose(1, 2, 0).squeeze(), cmap="gray"
            )
            plt.scatter(preds[img_num, :, 0], preds[img_num, :, 1], c="r", s=5)
            plt.scatter(landmarks[img_num, :, 0], landmarks[img_num, :, 1], c="g", s=5)

    print("Total number of test images: {}".format(len(valid_loader)))

    end_time = time.time()
    print("Elapsed Time : {}".format(end_time - start_time))


if __name__ == "__main__":

    model_path = "./saved/facial_keypoints/pth"
    device = "cpu"

    dataset = Dataset300W(transform=Transforms())
    _, valid_loader = get_train_val(dataset)

    evaluate(model_path, valid_loader, device)
