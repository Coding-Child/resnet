import gc
import os
import random
import logging
from tqdm import tqdm

import torch

import numpy as np
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt


losses_per_step = []
logging.basicConfig(filename="log.txt",
                    format='%(message)s',
                    filemode='w',
                    level=logging.INFO)


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def train_loop(device, model, data_loader, criterion, optimizer, scheduler=None, accumulation_steps=4):
    running_loss = 0.0

    model.train()
    optimizer.zero_grad()  # Move optimizer.zero_grad() up here

    with tqdm(total=len(data_loader), desc="Training") as pbar:
        for i, (img, label) in enumerate(data_loader):
            img = img.to(device)
            label = label.to(device).float()

            pred = model(img)
            loss = criterion(pred, label)
            loss = loss / accumulation_steps  # Normalize our loss (if averaged)
            loss.backward()

            if (i + 1) % accumulation_steps == 0 or i + 1 == len(data_loader):  # Wait for several backward steps
                optimizer.step()  # Now we can do an optimizer step
                optimizer.zero_grad()  # Reset gradients tensors
                if scheduler is not None:
                    scheduler.step()

            running_loss += loss.item() * accumulation_steps  # Accumulate the correct loss
            global losses_per_step
            losses_per_step.append(loss.item() * accumulation_steps)

            del img, label, pred  # Delete variables
            gc.collect()  # Force garbage collection
            torch.cuda.empty_cache()
            pbar.update(1)


def validation_epoch(device, model, data_loader, criterion):
    running_loss = 0.0
    all_preds = list()
    all_labels = list()

    model.eval()
    with tqdm(total=len(data_loader), desc="Validation") as pbar:
        with torch.no_grad():
            for img, label in data_loader:
                img = img.to(device)
                label = label.to(device).float()

                pred = model(img)
                loss = criterion(pred, label)

                running_loss += loss.item()

                all_preds.extend(pred.cpu().numpy().tolist())
                all_labels.extend(label.cpu().numpy().tolist())

                pbar.update(1)

    avg_val_loss = running_loss / len(data_loader)
    auc = roc_auc_score(all_labels, all_preds)

    return avg_val_loss, auc


def main(num_epochs, device, model, train_loader, dev_loader, test_loader,
         criterion, optimizer, scheduler_epoch=None, scheduler_step=None, save_path='model_check_point',
         accumulation_steps=8):
    min_loss = 1.0

    for epoch in range(num_epochs):
        train_loop(device, model, train_loader, criterion, optimizer, scheduler=scheduler_step,
                   accumulation_steps=accumulation_steps)
        valid_loss, valid_auc = validation_epoch(device, model, dev_loader, criterion)

        if scheduler_epoch is not None:
            scheduler_epoch.step(valid_loss)

        print(f"Epoch {epoch + 1}, Validation Loss: {valid_loss:.4f}, Validation AUC: {valid_auc:.4f}")

        logging.info(f"Epoch {epoch + 1}")
        logging.info(f"Validation AUC: {valid_auc:.4f}")
        logging.info(f"Validation Loss: {valid_loss:.4f}")

        if valid_loss < min_loss:
            print(f"Validation Loss decreased from {min_loss:.4f} to {valid_loss:.4f}. Saving model ...")
            os.makedirs('model_check_point', exist_ok=True)
            torch.save(model.state_dict(), save_path + "/best_model.pt")
            min_loss = valid_loss

    plt.plot(losses_per_step)
    plt.title("Training loss per step")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.show()

    test_auc_final, test_loss_final = validation_epoch(device, model, test_loader, criterion)
    model.load_state_dict(torch.load(save_path + '/best_model.pt'))
    test_auc_best, test_loss_best = validation_epoch(device, model, test_loader, criterion)

    print(f'Best Model AUC: {test_auc_best * 100:.2f}% & Final Model AUC: {test_auc_final * 100:.2f}%')
