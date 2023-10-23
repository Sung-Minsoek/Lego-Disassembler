import time
import torch
import torch.nn as nn

import test_GPT as test

# for early stopping and save model.
save_path = '../'

# for visualize.
train_accs = []
valid_accs = []


def train(model, trainloader, validloader, params, early_stopping=False):
    best_valid_acc = 0
    train_total = 0
    train_correct = 0

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

            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            running_loss += loss.item()
            num_samples += inputs.size(0)

            _, predicted = torch.max(output.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

            # Time
            end_time = time.time()
            time_taken = end_time - start_time
            time_taken = str(time_taken/60).split('.')

        valid_acc = test.test_with_labels_for_classification(model, validloader)

        if valid_acc > best_valid_acc and early_stopping == True:
            best_valid_acc = valid_acc

            checkpoint = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': params['optimizer'].state_dict(),
                'scheduler_state_dict': params['scheduler'].state_dict(),
            }

            torch.save(checkpoint, save_path)

        train_accs.append(train_correct / train_total * 100)
        valid_accs.append(valid_acc * 100)

        print('Epoch: {}/{}, train_acc: {:.2f}%, valid_acc: {:.2f}%, time:{}m {}s'.format(epoch + 1, params['epochs'], train_correct / train_total * 100, valid_acc * 100, time_taken[0], time_taken[1][:2]))

    return model
