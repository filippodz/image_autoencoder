# Model for images understanging: encode and decode images with Neural Network Autoencoder

Train an autoencoder able to encode and decode images

1. [Project Structure](#project-structure)
2. [Install](#install)
3. [Data preparing](#data-preparing)
4. [Train and Evaluate](#train-and-evaluate)

## Project Structure

```
$image_autoencoder
    |──models
        |──*.pth                # model weights    
    |──model.py                 # structure of the model
    |──download_data.py         # get dataset
    |──train.py                 # train model
    |──gen_reconstruction.py    # use model to encode and decode examples images to see the difference
    |──requirements.txt
    |──README.md
```

## Install


1. Clone the project
```shell
git clone https://github.com/filippodz/image_autoencoder.git
cd image_autoencoder
```
2. Install dependencies
```shell
pip install -r requirements.txt
```

## Data Preparing

```shell
python3 download_data.py
```

By running the download_data.py script you will obtain a dataset that will look like:

```
$dataset
    |──train
        |──class1
            |──xxxx.jpg
            |──...
        |──class2
            |──xxxx.jpg
            |──...
        |──...
        |──classN
            |──xxxx.jpg
            |──...
    |──val
        |──class1
            |──xxxx.jpg
            |──...
        |──class2
            |──xxxx.jpg
            |──...
        |──...
        |──classN
            |──xxxx.jpg
            |──...
```

## Train and Evaluate

For training just run the train.py script, the evolution of the train losses will show up in console and at the end into the train_losses.png plot.

For evaluating the capabilities of the model run the gen_reconstruction.py script and see the results in the reconstruction.jpg image.

```shell
python3 train.py
python3 gen_reconstruction.py 
```