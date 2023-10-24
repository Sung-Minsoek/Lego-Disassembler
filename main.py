import torch

from utils import dataloader, visualize
from utils import params as p
from train import train_GPT, train_Reg
from model import resnet3D

data_GPT_path = 'data/data_GPT/data_GPT.npy'
labels_GPT_path = 'data/data_GPT/labels_GPT.csv'
data_Reg_path = 'data/data_Reg/data_Reg.npy'
labels_Reg_path = 'data/data_Reg/labels_Reg.csv'

params = p.params

if __name__ == '__main__':
    # Load data
    data_GPT, labels_GPT = dataloader.data_and_labels_to_tensor(data_GPT_path, labels_GPT_path)
    data_Reg, labels_Reg = dataloader.data_and_labels_to_tensor(data_Reg_path, labels_Reg_path)

    trainloader_GPT, validloader_GPT = dataloader.make_dataloader(data_GPT, labels_GPT, params)
    trainloader_Reg, validloader_Reg = dataloader.make_dataloader(data_Reg, labels_Reg, params)

    # Visualize data
    visualize.show_img_GPT(trainloader_GPT)
    visualize.show_img_Reg(trainloader_GPT)

    # Generate model for Lego-GPT
    # Use ResNet-3D-50 network
    model_Lego_GPT = resnet3D.generate_model(50)

    # Initialization loss function, scheduler, optimizer
    p.init_train_setting(params, model_Lego_GPT, trainloader_GPT)

    # Train Lego-GPT
    model_Lego_GPT = train_GPT.train(model_Lego_GPT, trainloader_GPT, validloader_GPT, params, early_stopping=True)

    # Show result of Lego-GPT
    visualize.show_result(train_GPT.train_accs, train_GPT.valid_accs)
    visualize.show_prediction(model_Lego_GPT, validloader_GPT, params)

    # Generate model for Regression
    # Use ResNet-3D-50 network
    model_Reg = resnet3D.generate_model(50)

    # Initialization loss function, scheduler, optimizer
    p.init_train_setting(params, model_Reg, trainloader_Reg)

    # Load trained Lego-GPT network to Regression model
    checkpoint = torch.load(train_GPT.save_path)
    model_Reg.load_state_dict(checkpoint['model_state_dict'])

    # Start fine-tuning total network to Regression problem
    model_Reg = train_Reg.train(model_Reg, trainloader_Reg, validloader_Reg, params, early_stopping=True)

    # Show result of Regression
    visualize.show_result(train_Reg.train_losses, train_Reg.valid_losses)
