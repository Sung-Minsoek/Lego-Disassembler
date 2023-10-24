import time
import math
import torch
import torch.nn as nn


def test_for_regression(model, testloader, params):
    total_loss = 0.0
    num_samples = 0

    criterion = params['loss_function']
    model.eval()

    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(params['device']), data[1].to(params['device'])
            outputs = model(images)

            loss = criterion(outputs, labels)

            total_loss += loss * images.size(0)
            num_samples += images.size(0)

    return total_loss / num_samples
