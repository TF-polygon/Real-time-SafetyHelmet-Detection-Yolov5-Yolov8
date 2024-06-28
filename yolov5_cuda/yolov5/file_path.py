import os
import yaml
from glob import glob

from utils import dataloaders

root_path = '' #path of root

# Setting path
file_path = os.path.join(root_path, 'images/train')
valid_path = os.path.join(root_path, 'images/val')

#print('file path : ', file_path)
if os.path.exists(root_path):
    # Write all path of training images in train.txt
    with open(os.path.join(root_path, 'train.txt'), 'w') as f:
        for root, dirs, files in os.walk(file_path):
            for file in files:
                if file.lower().endswith('.jpg'):
                    file_path = os.path.join(root, file)
                    f.write(file_path + '\n')
    # Write all path of validation images in valid.txt
    with open(os.path.join(root_path, 'valid.txt'), 'w') as f:
        for root, dirs, files in os.walk(valid_path):
            for file in files:
                if file.lower().endswith('.jpg'):
                    valid_path = os.path.join(root, file)
                    f.write(valid_path + '\n')

data_yaml_path = os.path.join(root_path, 'data.yaml')
with open(data_yaml_path, 'r') as file:
    data_yaml = yaml.safe_load(file)