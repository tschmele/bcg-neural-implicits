import os
import time

import torch as th
from progress.bar import Bar
from torch.utils.tensorboard import SummaryWriter

import dataset
import models

# -------- CONFIGURATION --------
# --- select training ('train') or test ('test') mode
c_mode = 'train'
c_model = models.SimpleModel                                                # which model class to use
c_loss_fn = th.nn.L1Loss                                                    # which loss function to use
c_dataset_n = '128k'                                                       # dataset size
c_storage_path = f'{c_dataset_n}-dataset/'                                  # which folder to save to
c_dataset_path = f'../dataset-generation/{c_storage_path}'                  # which dataset to use

# --- training options
c_initial_model_path = None                                         # path to initial model, to resume training from. None for training from scratch.            
c_batch_size = 1024                                                  # batch size
c_epochs = 100                                                      # number of epochs to train (number of times the model gets to see all training data)
c_learning_rate = 0.0001                                            # learning rate
c_loss_report_interval = 50                                         # report loss every n batches

# --- Evaluation options
c_test_model_path = 'NN.pth'                                       # path to model snapshot which should be used for testing

# --- Evaluation callback for drawing
# Takes an image and the model and returns the class probabilities.
# The image must be sampled down to the input image shape of the model
# and batch and channel dimensions must be added. Look up the documentation of
# the tensor methods 'squeeze' and 'unsqueeze' to add and remove dimensions.
# For conversion from and to numpy, the functions 'numpy' and 'from_numpy' can be used.
# We don't need gradients, so use 'with th.no_grad():' to disable them.
# To get actual probabilities, apply softmax to the output of the model.

# --- Training step for one batch
# Performs one training step for a batch of training data, given a model, an optimizer and a loss function.
# Returns the loss of the batch.
def train_step(model, optimizer, loss_fn, batch):
    # Unpack input and target from batch
    x, y_hat = batch
    # Reset the gradients
    optimizer.zero_grad()
    # Forward pass
    y = model(x)
    # Calculate the loss
    loss = loss_fn(y, y_hat)
    # Backward pass (calculate gradients w.r.t. model parameters)
    loss.backward()
    # Update the model parameters by performing a step in the negative gradient direction (the amount depends on the optimizer and the learning rate)
    optimizer.step()
    # Return loss and network output
    return loss.item(), y

# --- Validation step for one batch
# Performs one validation step for a batch of validation data, given a model and a loss function.
# Returns the loss of the batch and the network output for calculating accuracy.
def validation_step(model, loss_fn, batch):
    # Unpack input and target from batch
    x, y_hat = batch
    # Forward pass
    y = model(x)
    # Calculate the loss
    loss = loss_fn(y, y_hat)
    # Return loss and network output
    return loss.item(), y

# --- MAIN FUNCTION ---
def main():
    # --- MODEL DEFINITION

    # Create a model instance
    # We need to specify the input shape and the number of classes
    model = c_model()

    # Define the loss function (cross entropy loss is a good choice for classification)
    loss_fn = c_loss_fn()

    # print model summary
    print("=== Model Summary ===")
    print(model)

    # --- TRAINING MODE
    if c_mode == 'train':
        # --- Load training data
        # Load dataset, train and test
        testtrainingdata = dataset.CustomImageDataset(f'{c_dataset_path}samples.txt', f'{c_dataset_path}distances.txt')
        testtrainingdata_test = dataset.CustomImageDataset(f'{c_dataset_path}samples_test.txt', f'{c_dataset_path}distances_test.txt')

        # Create a dataloader for the training set
        train_loader = th.utils.data.DataLoader(testtrainingdata, batch_size=c_batch_size, shuffle=True)
        # Create a dataloader for the test set
        test_loader = th.utils.data.DataLoader(testtrainingdata_test, batch_size=c_batch_size, shuffle=True)

        # Hint: Data loaders are iterables, on each iteration they return a batch of data.
        # The batch is tuple of two tensors: the first tensor contains the input images of the batch (shape: [batch_size, channels, height, width]),
        # the second tensor contains the target labels of the batch (shape: [batch_size]).
        # There are two data loaders, one for the training set and one for the test set.

        # --- if initial model is given, load its weights and continue training from there
        if c_initial_model_path is not None:
            model.load_state_dict(th.load(c_initial_model_path))

        # --- Optimization loop. Looks scary but is mostly logging and bookkeeping. In each epoch, for each batch, we execute a training and validation step.
        # These are implemented in the functions train_step and validation_step above.

        # Define the optimizer (AdamW is a good first choice. Other typical choices are SGD or Adam)
        # We specify which parameters should be optimized (model.parameters()) and the learning rate.
        optimizer = th.optim.Adam(model.parameters(), lr=c_learning_rate)

        # Create tensorboard writer for logging
        writer = SummaryWriter()

        # Bookkeeping variables
        training_iteration = 0              # current total number of training batches processed
        validation_iteration = 0            # current total number of validation batches processed
        training_loss = 0.0                 # current training loss averaged over c_loss_report_interval batches
        validation_loss = 0.0               # current validation loss averaged over c_loss_report_interval batches
        training_epoch_loss = 0.0           # current training loss averaged over all batches in the current epoch
        validation_epoch_loss = 0.0         # current validation loss averaged over all batches in the current epoch
        training_positives = 0              # current number of correctly classified training samples out of the current epoch
        validation_positives = 0            # current number of correctly classified validation samples out of the current epoch

        # Loop over epochs
        for epoch in range(c_epochs):
            print(f"=== Epoch {epoch + 1}/{c_epochs} ===")

            # reset epoch losses
            training_epoch_loss = 0.0
            validation_epoch_loss = 0.0
            training_positives = 0
            validation_positives = 0

            # --- TRAINING
            # Enable training mode
            model.train()
            # Progress bar
            t0 = time.time()
            pgb = Bar("[TRAINING]", max = len(train_loader), suffix = 'batch %(index)d/%(max)d - %(percent)d%% - eta: %(eta)ds')

            # Loop over batches
            for batch_idx, batch in enumerate(train_loader):
                # Train step
                loss, output = train_step(model, optimizer, loss_fn, batch)

                # Accumulate losses
                training_loss += loss
                training_epoch_loss += loss

                # Count positives for accuracy calculation
                training_positives += th.sum(th.argmax(output, dim=0) == batch[1]).item()

                # Report average loss every c_loss_report_interval batches (tensorboard)
                if (training_iteration + 1) % c_loss_report_interval == 0:
                    writer.add_scalar('iteration/training_loss', training_loss / c_loss_report_interval, epoch * len(train_loader) + batch_idx + 1)
                    training_loss = 0.0
                training_iteration += 1
                pgb.next()

            # Print training time
            t1 = time.time()
            print(f"\n-> done! {((t1 - t0) * 1000.0):.2f} ms")

            # Log avg. training loss over epoch
            writer.add_scalar('epoch/training_loss', training_epoch_loss / len(train_loader), epoch + 1)
            print(f"Training loss: {training_epoch_loss / len(train_loader):.4f}")

            # Log avg. training accuracy over epoch
            writer.add_scalar('epoch/training_accuracy', training_positives / len(testtrainingdata), epoch + 1)
            print(f"Training accuracy: {training_positives / len(testtrainingdata):.4f}")

            # --- VALIDATION
            # Enable evaluation mode
            model.eval()
            with th.no_grad(): # We don't need gradients for validation
                t0 = time.time()
                # Progress bar
                pgb = Bar("[VALIDATION]", max = len(test_loader), suffix = 'batch %(index)d/%(max)d - %(percent)d%% - eta: %(eta)ds')
                for batch_idx, batch in enumerate(test_loader):
                    # Validation step
                    loss, output = validation_step(model, loss_fn, batch)

                    # Accumulate losses
                    validation_loss += loss
                    validation_epoch_loss += loss

                    # Count positives for accuracy calculation
                    validation_positives += th.sum(th.argmax(output, dim=0) == batch[1]).item()

                    # Report average loss every c_loss_report_interval batches (tensorboard)
                    if (validation_iteration + 1) % c_loss_report_interval == 0:
                        writer.add_scalar('iteration/validation_loss', validation_loss / c_loss_report_interval, epoch * len(test_loader) + batch_idx + 1)
                        validation_loss = 0.0
                    validation_iteration += 1
                    pgb.next()

                # Print validation time
                t1 = time.time()
                print(f"\n-> done! {((t1 - t0) * 1000.0):.2f} ms")

                # Log avg. validation loss over epoch
                writer.add_scalar('epoch/validation_loss', validation_epoch_loss / len(test_loader), epoch + 1)
                print(f"Validation loss: {validation_epoch_loss / len(test_loader):.4f}")

                # Log avg. validation accuracy over epoch
                writer.add_scalar('epoch/validation_accuracy', validation_positives / len(testtrainingdata_test), epoch + 1)
                print(f"Validation accuracy: {validation_positives / len(testtrainingdata_test):.4f}")

            # Save model snapshot every epoch
            if not os.path.isdir(c_storage_path):
                os.makedirs(c_storage_path)
            th.save(model.state_dict(), f"{c_storage_path}{model.__class__.__name__}_epoch_{epoch + 1}.pth")

        # Close tensorboard writer
        writer.close()

    # --- TEST MODE
    elif c_mode == 'test':
        # Load the model
        model.load_state_dict(th.load(c_test_model_path))
        # Enable evaluation mode
        model.eval()

# call main function if this file is executed
if __name__ == "__main__":
    main()
