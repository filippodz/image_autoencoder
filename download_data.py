import shutil
from pathlib import Path
from fastai.vision.all import untar_data, URLs

def filter_and_rename_classes(base_path, classes_to_keep):
    for class_id, new_class_name in classes_to_keep.items():
        original_class_path = base_path / class_id
        new_class_path = base_path / new_class_name
        if original_class_path.exists():
            original_class_path.rename(new_class_path)
    for class_path in base_path.iterdir():
        if class_path.name not in classes_to_keep.values():
          try:
            shutil.rmtree(class_path)
          except:
            pass

if __name__ == "__main__":
  dataset_path = Path.cwd()/"dataset"
  classes_to_keep = {
      'n03028079': 'church',
      'n03394916': 'horn',
      'n03445777': 'golf ball',
  }

  if not (dataset_path).exists():
    path = untar_data(URLs.IMAGENETTE_160)
    shutil.move(path, dataset_path)
    train_path = dataset_path / 'train'
    valid_path = dataset_path / 'val'
    filter_and_rename_classes(train_path, classes_to_keep)
    filter_and_rename_classes(valid_path, classes_to_keep)
  print(f"Downloaded dataset to: {dataset_path}")