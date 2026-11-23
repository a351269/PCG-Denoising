"""
PyTorch port of network_training.py (Keras / TensorFlow 2.3).

Data layout is channels-first: (N, 1, H, W).
"""
import os
import json
import h5py
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import pcg_denoising.utils.network as net


"""
Data functions
"""
def load_h5(path):
    with h5py.File(path, "r") as f:
        return f["mag_noisy"][:], f["mag_target"][:]

def resize(x, y):
    # Drop the last frequency bin / time frame if the size is odd
    if np.shape(x)[1] % 2 != 0:
        x = x[:, :-1, :]
        y = y[:, :-1, :]

    if np.shape(x)[2] % 2 != 0:
        x = x[:, :, :-1]
        y = y[:, :, :-1]

    return x, y

def add_channel(a):
    # (N, H, W) -> (N, 1, H, W)
    return np.ascontiguousarray(a[:, np.newaxis], dtype=np.float32)

def prepare_datasets(fold_dir):
    """fold_dir holds train.h5 and val.h5. Returns (x_train, y_train, x_val, y_val)."""
    fold_dir = Path(fold_dir)
    x_train, y_train = resize(*load_h5(fold_dir / "train.h5"))
    x_val, y_val = resize(*load_h5(fold_dir / "val.h5"))
    return add_channel(x_train), add_channel(y_train), add_channel(x_val), add_channel(y_val)


"""
Metrics / epoch loop
"""
def psnr_from_mse(mse, max_pixel=1.0):
    mse = max(mse, 1e-12)
    return 20.0 * np.log10(max_pixel / np.sqrt(mse))


def run_epoch(model, loader, device, optimizer=None):
    """One pass over `loader`. Trains if an optimizer is given, otherwise evaluates.
    Returns (mae, psnr)."""
    training = optimizer is not None
    model.train(training)
    n, abs_sum, sq_sum = 0, 0.0, 0.0

    with torch.set_grad_enabled(training):
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            pred = model(xb)               # (B, 2, H, W)
            target = yb.expand_as(pred)    # Keras broadcast the 1-channel target to both outputs
            loss = F.l1_loss(pred, target)

            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            count = pred.numel()
            abs_sum += loss.item() * count
            sq_sum += F.mse_loss(pred.detach(), target).item() * count
            n += count

    mae = abs_sum / n
    return mae, psnr_from_mse(sq_sum / n)


"""
Model training
"""
def unetTraining(tf_path, model_path, model_history, lr, batch, epochs, seed):
    if os.path.exists(model_path):
        os.remove(model_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print("Device:", device)
    torch.backends.cudnn.benchmark = True

    x_train, y_train, x_validation, y_validation = prepare_datasets(tf_path)
    print(x_train.shape, y_train.shape, x_validation.shape, y_validation.shape, np.min(x_train), np.max(x_train), np.min(y_train), np.max(y_train))

    def make_loader(x, y, shuffle):
        ds = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
        return DataLoader(ds, batch_size=batch, shuffle=shuffle, pin_memory=(device.type == "cuda"))

    train_loader = make_loader(x_train, y_train, shuffle=True)
    val_loader = make_loader(x_validation, y_validation, shuffle=False)

    torch.manual_seed(seed)
    model = net.UNet().to(device)
    print(model)
    print("Parameters:", sum(p.numel() for p in model.parameters()))

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, eps=1e-7)

    history = {"mae": [], "val_mae": [], "psnr": [], "val_psnr": []}
    bar = tqdm(range(1, epochs + 1), desc="Training", unit="epoch")
    for epoch in bar:
        mae, psnr = run_epoch(model, train_loader, device, optimizer)
        val_mae, val_psnr = run_epoch(model, val_loader, device)

        for key, value in dict(mae=mae, val_mae=val_mae, psnr=psnr, val_psnr=val_psnr).items():
            history[key].append(value)

        bar.set_postfix(mae=f"{mae:.5f}", psnr=f"{psnr:.2f}", val_mae=f"{val_mae:.5f}", val_psnr=f"{val_psnr:.2f}")

    torch.save({"state_dict": model.state_dict()}, model_path)

    with open(model_history, "w") as f:
        json.dump(history, f)

    return model, history


"""
Prediction functions
"""
def load_model(path, device=None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = net.UNet()
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    return model.to(device).eval()


def Predict(model, x, batch_size=32):
    """x: (N, H, W) -> prediction: (N, H, W, 2), channels-last like the Keras model."""
    device = next(model.parameters()).device
    model.eval()
    x_t = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))[:, None]  # (N, 1, H, W)
    out = []
    with torch.no_grad():
        for i in range(0, len(x_t), batch_size):
            out.append(model(x_t[i:i + batch_size].to(device)).cpu())
    return torch.cat(out).permute(0, 2, 3, 1).numpy()


def Resize_Network(x):
    """Drop a trailing odd row / column. Returns (x, temp1) where temp1 is the
    removed last row (or None if nothing was removed along axis 1)."""
    temp1 = None
    if np.shape(x)[1] % 2 != 0:
        temp1 = x[:, -1, :]
        x = x[:, :-1, :]

    if np.shape(x)[2] % 2 != 0:
        x = x[:, :, :-1]

    return x, temp1


def Resize_Network_Inverse(x, temp1):  # x[16, 128, 64], temp1[16, 64]
    return np.concatenate([x, temp1[:, np.newaxis, :]], axis=1)  # -> [16, 129, 64]