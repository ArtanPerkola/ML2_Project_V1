import os
import shutil
import random

def split_data(source, train, val, split_size):
    all_files = os.listdir(source)
    random.shuffle(all_files)
    split_index = int(len(all_files) * split_size)
    train_files = all_files[split_index:]
    val_files = all_files[:split_index]
    
    for file in train_files:
        shutil.move(os.path.join(source, file), os.path.join(train, file))
        
    for file in val_files:
        shutil.move(os.path.join(source, file), os.path.join(val, file))

base_dir = 'data'
categories = ['benign', 'malignant']
split_size = 0.2

for category in categories:
    source = os.path.join(base_dir, 'train', category)
    train_dest = os.path.join(base_dir, 'train', category)
    val_dest = os.path.join(base_dir, 'validation', category)
    
    os.makedirs(val_dest, exist_ok=True)
    
    split_data(source, train_dest, val_dest, split_size)

print("Data split complete.")
