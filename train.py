"""
train.py
==================
This script trains a neural network model for the Battleship game using self-play data.

"""

import json
import torch 
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tfg.ai.network import GameNet

class GameDataset(Dataset):
    """GameDataset is a PyTorch Dataset for loading Battleship game records from a JSONL file.

    Args:
        path (str): Path to the JSONL file containing game records.
    """
    def __init__(self, path):
        """ Initializes the dataset by loading records from a JSONL file.

        Args:
            path (str): Path to the JSONL file containing game records.
        """
        with open(path) as f:
            self.data = [json.loads(l) for l in f]

    def __len__(self):
        """ Returns the number of records in the dataset.

        Returns:
            int: The number of records in the dataset.
        """
        return len(self.data)

    def __getitem__(self, idx):
        """ Retrieves a single record from the dataset by index.

        Args:
            idx (int): Index of the record to retrieve.

        Returns:
            tuple: A tuple containing:
                - x (torch.Tensor): The state tensor of shape (3, 6, 6).
                - pi (torch.Tensor): The policy distribution tensor of shape (36,).
                - z (torch.Tensor): The outcome value tensor of shape (1,).
        """
        rec = self.data[idx]
        x  = torch.tensor(rec['state'], dtype=torch.float32)
        pi = torch.tensor(rec['pi'],    dtype=torch.float32)
        z  = torch.tensor(rec['z'],     dtype=torch.float32)
        return x, pi, z

def train(
    datafile='data_balanced.jsonl',
    epochs=50,
    batch_size=64,
    lr=1e-3,
    model_out='model.pth',
    val_split=0.1,
    early_stop_patience=5
):
    """ Train a neural network model for the Battleship game using self-play data.

    Args:
        datafile (str): Path to the JSONL file containing game records.
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.
        lr (float): Learning rate for the optimizer.
        model_out (str): Output path to save the trained model.
        val_split (float): Fraction of data to use for validation.
        early_stop_patience (int): Number of epochs with no improvement before stopping training early.
    """
    # DS and split
    ds = GameDataset(datafile)
    n_val = int(len(ds) * val_split)
    n_trn = len(ds) - n_val
    trn_ds, val_ds = random_split(ds, [n_trn, n_val])
    trn_loader = DataLoader(trn_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # Model, optimizer, scheduler
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GameNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min',
                                  factor=0.5, patience=3,
                                  verbose=True)

    best_val_loss = float('inf')
    epochs_no_improve = 0

    for ep in range(1, epochs+1):
        # training
        model.train()
        total_train_loss = 0
        for x, pi, z in trn_loader:
            x, pi, z = x.to(device), pi.to(device), z.to(device)
            logits, v = model(x)

            # Policy loss: cross‐entropy vs pi target
            loss_p = - (pi * F.log_softmax(logits, dim=1)).sum(dim=1).mean()
            # Value loss: MSE vs z
            loss_v = F.mse_loss(v.squeeze(-1), z)
            loss = loss_p + loss_v

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(trn_loader)

        # validation
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for x, pi, z in val_loader:
                x, pi, z = x.to(device), pi.to(device), z.to(device)
                logits, v = model(x)
                loss_p = - (pi * F.log_softmax(logits, dim=1)).sum(dim=1).mean()
                loss_v = F.mse_loss(v.squeeze(-1), z)
                total_val_loss += (loss_p + loss_v).item()
        avg_val_loss = total_val_loss / len(val_loader)

        # Scheduler on validation loss
        scheduler.step(avg_val_loss)

        print(f"Epoch {ep}/{epochs} — "
              f"Train Loss: {avg_train_loss:.4f} — "
              f"Val Loss: {avg_val_loss:.4f}")

        # Early stopping + best‐model saving
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), model_out)
            print(f"New best model saved (val loss {best_val_loss:.4f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= early_stop_patience:
                print(f"Early stopping: no improvement in {early_stop_patience} epochs.")
                break

    print("Training completed.")

if __name__ == '__main__':
    train()
