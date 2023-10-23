import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F


# show_img series:
# Just pick five tensors randomly, show image of data.
def show_img_GPT(dataloader):
    dataiter = iter(dataloader)
    videos, labels = next(dataiter)

    for i in range(5):
        video = videos[i, 0, :, :, :]
        label = labels[i]

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        plt.subplots_adjust(wspace=0.1)

        frame1 = video[0, :, :]
        frame2 = video[1, :, :]

        axes[0].imshow(frame1.squeeze(), cmap='gray')
        axes[0].axis('off')
        axes[0].set_title(f'Frame {0} Label {label}')

        axes[1].imshow(frame2.squeeze(), cmap='gray')
        axes[1].axis('off')
        axes[1].set_title(f'Frame {1} Label {label}')

        plt.show()

def show_img_Reg(dataloader):
    dataiter = iter(dataloader)
    videos, labels = next(dataiter)

    for i in range(5):
        video = videos[i, 0, :, :, :]
        label = labels[i]

        fig, axes = plt.subplots(1, 4, figsize=(10, 4))
        plt.subplots_adjust(wspace=0.1)

        for f in range(4):
            frame = video[f, :, :]

            axes[f].imshow(frame.squeeze(), cmap='gray')
            axes[f].axis('off')
            axes[f].set_title(f'Frame {f}')

        plt.title(f'Label{label}')
        plt.show()


# show_result series:
# Show graph of train and validation accuracy(GPT) or loss(Reg).
def show_result(train_result: list, valid_result: list):
    plt.plot(train_result, label='Train')
    plt.plot(valid_result, label='Validation')

    plt.title('Validation Accuracy of Lego-GPT')
    plt.xlabel('accuracy')
    plt.ylabel('epochs')

    plt.legend()
    plt.show()


# show_prediction:
# Show input images and model predictions.
def show_prediction(model, validloader, params):
    dataiter = iter(validloader)
    videos, labels = next(dataiter)

    classes = ['2x4', '2x3', '2x2', '1x2', '1x1', '1x4', '1x3']

    for i in range(4):
        video = videos[i, 0, :, :, :]
        label = labels[i]

        fig, axes = plt.subplots(1, 3, figsize=(10, 4))
        plt.subplots_adjust(wspace=0.1)

        frame1 = video[0, :, :]
        frame2 = video[1, :, :]

        axes[0].imshow(frame1.squeeze(), cmap='gray')
        axes[0].axis('off')
        axes[0].set_title(f'View {1} Label {classes[label]}')

        axes[1].imshow(frame2.squeeze(), cmap='gray')
        axes[1].axis('off')
        axes[1].set_title(f'View {2} Label {classes[label]}')

        logit = model(videos.to(params['device']))
        prob = F.softmax(logit, dim=1)
        prob = prob.detach().cpu().squeeze().tolist()

        axes[2].bar(classes, prob[i])
        axes[2].set_title(f'Prediction')

        plt.show()