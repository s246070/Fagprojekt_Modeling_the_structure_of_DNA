from dna_modelling.model import Model
from dna_modelling.data import MyDataset

def TrainModel(model,train_dataloader, epochs=1,val_dataloader=None, device=None):

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
