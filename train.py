import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from pathlib import Path

from model import *

from fastai.vision.all import set_seed, DataBlock, ImageBlock, get_image_files, RandomSplitter, Resize, Learner

def get_y(x):
    return x

if __name__ == "__main__":
    set_seed(22)
    dataset_path = Path.cwd()/"dataset"
    batch_size=10
    img_resize_dim=256

    data_block = DataBlock(
    blocks=(ImageBlock, ImageBlock),
    get_items=get_image_files,
    splitter=RandomSplitter(valid_pct=0.2),
    get_y=get_y,
    item_tfms=Resize(img_resize_dim),
    )

    dls = data_block.dataloaders(dataset_path / "train", bs=batch_size)

    autoencoder_model = Autoencoder()

    learn = Learner(dls, autoencoder_model, loss_func = nn.MSELoss())
    model_name = "Autoencoder"

    try:
        learn.load(model_name)
        print("Model already exists, training not performed")
    except:
        learn.fit_one_cycle(20)
        learn.save(model_name)
        plt.figure(figsize=(10, 5))
        learn.recorder.plot_loss()
        plt.title('Training Losses')
        plt.savefig('training_losses.png')
        print("Training ended")