from dna_modelling.model import Model
from dna_modelling.data import Data
import torch
import torch.nn as nn




def TrainModel(model,train_dataloader, epochs=1,val_dataloader=None, device=None, threads=1, loss_fn=ordinal_cross_entropy_loss(), optimizer= Adam()):
    """Trains the model for a given number of epochs.
    
    Args:
        model: The Latent distance model to be trained.
        train_dataloader: The dataloader for the training data.
        epochs: The number of epochs to train for.
        val_dataloader: The dataloader for the validation data (optional).
        device: The device to train on (optional).
        threads: The number of threads to use for training (optional).
    
    Returns:
        A tuple containing the training accuracy, validation accuracy, training loss, and validation loss.
    """
    
    
    torch.set_number_of_threads(threads) # brug denne til at sætte antal tråde.
    model.train()

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model.to(device)

    train_acc = []
    test_acc = []
    train_loss = []
    val_loss = []

    for epoch in range(epochs):
        epoch_train_loss = 0
        epoch_train_acc = 0

        for batch in train_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            loss, acc = model.training_step(batch)
            epoch_train_loss += loss.item()
            epoch_train_acc += acc.item()

        train_acc.append(epoch_train_acc / len(train_dataloader))
        train_loss.append(epoch_train_loss / len(train_dataloader))

        if val_dataloader is not None:
            epoch_val_loss = 0
            epoch_val_acc = 0

            with torch.no_grad():
                for batch in val_dataloader:
                    batch = {k: v.to(device) for k, v in batch.items()}
                    loss, acc = model.validation_step(batch)
                    epoch_val_loss += loss.item()
                    epoch_val_acc += acc.item()

            test_acc.append(epoch_val_acc / len(val_dataloader))
            val_loss.append(epoch_val_loss / len(val_dataloader))