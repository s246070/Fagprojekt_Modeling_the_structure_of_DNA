from dna_modelling.model import Model
from dna_modelling.data import MyDataset

def TrainModel(model,train_dataloader, epochs=1,val_dataloader=None, device=None):
    """Trains the model for a given number of epochs.
    
    Args:
        model: The model to be trained.
        train_dataloader: The dataloader for the training data.
        epochs: The number of epochs to train for.
        val_dataloader: The dataloader for the validation data (optional).
        device: The device to train on (optional).
    
    Returns:
        A tuple containing the training accuracy, validation accuracy, training loss, and validation loss.
    """
    
    
    torch.set_number_of_threads(1) # brug denne til at sætte antal tråde.
    model.train()

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model.to(device)

    train_acc = []
    test_acc = []
    train_loss = []
    val_loss = []

    for epoch in range(epochs):





    

if __name__ == "__main__":
    train()
