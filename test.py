import time
import torch
import torch.nn as nn

def test_with_labels_for_classification(model, testloader):
    val_correct = 0
    val_total = 0

    model.eval()

    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    return val_correct / val_total

def test_without_labels_for_classification(model, testloader):
    answer = []

    model.eval()

    with torch.no_grad():
        for data in testloader:
            images = data.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            answer.extend(predicted.tolist())

    return answer
