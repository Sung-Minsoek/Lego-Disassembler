def train(model, trainloader, validloader, params):

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

        valid_loss = test_for_regression(model, validloader)

        print('Epoch: {}/{}, train_loss: {:.4f}, valid_loss: {:.2f}, time:{}m {}s'.format(epoch + 1, params['epochs'], running_loss / num_samples, valid_loss, time_taken[0], time_taken[1][:2]))
