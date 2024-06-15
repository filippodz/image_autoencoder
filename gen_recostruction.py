import torch
import torch.nn as nn
from torchvision.transforms import transforms
from matplotlib import pyplot as plt
from pathlib import Path
from tqdm import tqdm

from model import *

from fastai.vision.all import DataBlock, ImageBlock, get_image_files, RandomSplitter, Resize, Learner

def get_y(x):
    return x

if __name__ == "__main__":
    dataset_path = Path.cwd()/"dataset"
    batch_size=1
    img_resize_dim=256

    data_block = DataBlock(
    blocks=(ImageBlock, ImageBlock),
    get_items=get_image_files,
    splitter=RandomSplitter(valid_pct=0.2),
    get_y=get_y,
    item_tfms=Resize(img_resize_dim),
    )

    dls = data_block.dataloaders(dataset_path / "val", bs=batch_size)

    autoencoder_model = Autoencoder()

    learn = Learner(dls, autoencoder_model, loss_func = nn.MSELoss())
    model_name = "Autoencoder"

    learn.load(model_name)

    print("Model loaded")

    plt.figure(figsize=(16, 9))
    for i, (input, target) in tqdm(enumerate(dls.valid)):
        output = learn.model(input)
        input = transforms.ToPILImage()(input.squeeze().cpu())
        output = transforms.ToPILImage()(output.squeeze().cpu())
        plt.subplot(6,12,2*i+1, xticks=[], yticks=[])
        plt.imshow(input)

        plt.subplot(6,12,2*i+2, xticks=[], yticks=[])
        plt.imshow(output)

        if i == 35:
            break
    plt.savefig('reconstruction.jpg')
    print(f"Saved example of model recontruction: {Path.cwd()/'reconstruction.jpg'}")