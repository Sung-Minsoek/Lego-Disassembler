from utils import dataloader, params, visualize
from train import train_GPT
from model import resnet3D

data_GPT_path = 'data/data_GPT/data_GPT.npy'
labels_GPT_path = 'data/data_GPT/labels_GPT.csv'
data_Reg_path = 'data/data_Reg/data_Reg.npy'
labels_Reg_path = 'data/data_Reg/labels_Reg.csv'

params = params.params

if __name__ == '__main__':
    # Initialization parameters
    params.init_params(params)

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
    model = resnet3D.generate_model(50)

    # Train Lego-GPT
    model = train_GPT.train(model, trainloader_GPT, validloader_GPT, params, early_stopping=True)

    # Show result
    visualize.show_result(train_GPT.train_accs, train_GPT.valid_accs)
    visualize.show_prediction(model, validloader_GPT, params)
