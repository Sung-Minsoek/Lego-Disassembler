import time
import torch
import torch.nn as nn

import test_Reg as test

# for early stopping and save model.
save_path = './best_model_Reg.pth'

# for visualize.
train_losses = []
valid_losses = []


def train(model, trainloader, validloader, params, early_stopping=False):
    best_valid_loss = float("inf")

    device = params['device']

    optimizer = params['optimizer']
    criterion = params['loss_function']
    scheduler = params['scheduler']

    model.train()

    for epoch in range(params['epochs']):
        running_loss = 0.0
        num_samples = 0

        start_time = time.time()

        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast(True):
                output = model(inputs)
              
            output = output.type(torch.float32)

            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            running_loss += loss.item()
            num_samples += inputs.size(0)

            #Time
            end_time = time.time()
            time_taken = end_time - start_time
            time_taken = str(time_taken/60).split('.')

        valid_loss = test.test_for_regression(model, validloader)

        if valid_loss < best_valid_loss and early_stopping:
            best_valid_loss = valid_loss

            checkpoint = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': params['optimizer'].state_dict(),
                'scheduler_state_dict': params['scheduler'].state_dict(),
            }

            torch.save(checkpoint, save_path)

        train_losses.append(running_loss / num_samples)
        valid_losses.append(valid_loss)

        print('Epoch: {}/{}, train_loss: {:.4f}, valid_loss: {:.4f}, time:{}m {}s'.format(epoch + 1, params['epochs'], running_loss / num_samples, valid_loss, time_taken[0], time_taken[1][:2]))
