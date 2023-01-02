import numpy as np
from util import *
from cfg import get_cfg
from tqdm.auto import tqdm
import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib as mpl
import matplotlib.pyplot as plt
from models import get_model

# mpl.use("TkAgg")

def train(model, train_loader, optimizer, scheduler=None):
    model.train()
    if cfg.tqdm:
        tqdm_bar = tqdm(train_loader)
    else:
        tqdm_bar = train_loader
    for image, label in tqdm_bar:
        image = image.to(DEVICE)
        label = label.to(DEVICE)
        optimizer.zero_grad()
        output = model(image)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        if cfg.tqdm:
            tqdm_bar.set_description(f"Epoch {epoch} - train loss: {loss.item():.6f}")

def evaluate(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        if cfg.tqdm:
            tqdm_bar = tqdm(test_loader)
        else:
            tqdm_bar = test_loader
        for image, label in tqdm_bar:
            image = image.to(DEVICE)
            label = label.to(DEVICE)
            output = model(image)
            test_loss += criterion(output, label).item()
            prediction = output.max(1, keepdim=True)[1]
            correct += prediction.eq(label.view_as(prediction)).sum().item()
    
    test_loss /= len(test_loader.dataset)
    test_accuracy = 100. * correct / len(test_loader.dataset)
    return test_loss, test_accuracy

if __name__ == "__main__":
    cfg = get_cfg()
    DEVICE = torch.device('cuda') if torch.cuda.is_available else torch.device('cpu')
    print(f"Using PyTorch version: {torch.__version__}, Device: {DEVICE}")
    
    wandb.init(project=cfg.wandb, config=cfg)

    ## define training vars
    EPOCHS = cfg.epoch
    model = get_model(cfg.name.lower())().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=cfg.factor)
    criterion = nn.CrossEntropyLoss()
    
    ## import dataset
    train_dataset = CustomDataset(train=True, prob=cfg.aug_p)
    test_dataset = CustomDataset(train=False, prob=0.0)

    train_loader = DataLoader(dataset=train_dataset, 
                            batch_size=cfg.bs, 
                            shuffle=True, 
                            collate_fn=train_dataset.collate_fn)
    test_loader = DataLoader(dataset=test_dataset, 
                            batch_size=cfg.bs, 
                            shuffle=False, 
                            collate_fn=test_dataset.collate_fn)
    
    ## training
    accs = np.zeros(shape=EPOCHS)
    losses = np.zeros(shape=EPOCHS)
    max_acc = 0
    saved_model = None
    fname = ""
    save_epoch = 0
    
    for epoch in range(EPOCHS):
        train(model, train_loader, optimizer, scheduler)
        test_loss, test_accuracy = evaluate(model, test_loader)
        print(f"\n[EPOCH: {epoch+1}], \tModel: {model.__class__.__name__}, \tTest Loss: {test_loss:.4f}, \tTest Accuracy: {test_accuracy:.2f} % \n")
        accs[epoch] = test_accuracy
        losses[epoch] = test_loss
        
        ## save point
        if max_acc < accs[epoch]:
            max_acc = accs[epoch]
            saved_model = model.state_dict()
            fname = f"./saved_models/{cfg.dataset}_{model.__class__.__name__}_{cfg.bs}.pt"
            save_epoch = epoch + 1
        else:
            torch.save(saved_model, fname)
            print(f"Model save at epoch {save_epoch}")
        
        wandb.log({"acc": test_accuracy}, commit=False)
        wandb.log({"loss": test_loss})
        scheduler.step(test_loss)
        
    ## plot save
    if cfg.save_plot:
        plt.figure(figsize=(15, 6))
        
        plt.subplot(121)
        plt.plot(losses)
        plt.title("Test loss")
        plt.grid(True)

        plt.subplot(122)
        plt.plot(accs)
        plt.title("Test accuracy")
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(f"plots/{cfg.dataset}_{model.__class__.__name__}_{cfg.bs}.png")