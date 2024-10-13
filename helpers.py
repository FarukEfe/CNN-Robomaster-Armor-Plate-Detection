import os
import shutil
from math import floor

def order_dataset(val_split: float = 0.2, test_split: float = 0.1):
    if val_split + test_split >= 1:
        print("Validation + Test cannot exceed 1")
        return
    
    if (os.path.exists("./train/pos")):
        return
    # Create new folders labelled Pos and Neg
    os.mkdir("./data/train/pos")
    os.mkdir("./data/train/neg")
    os.mkdir("./data/val")
    os.mkdir("./data/val/pos")
    os.mkdir("./data/val/neg")
    os.mkdir("./data/test")
    os.mkdir("./data/test/pos")
    os.mkdir("./data/test/neg")
    # Get validation index and test index

    # Add files based on label
    for dirname, _, filenames in os.walk(os.getcwd() + "./data/train/"):
        val_idx= floor(len(filenames) - len(filenames) * (val_split + test_split))
        test_idx = floor(len(filenames) - len(filenames) * test_split)
        # Iterate through each image in list with its subsequent txt
        for i in range(0,len(filenames) - 1,2):
            img_name = filenames[i]
            txt_name = filenames[i+1] # Get label in txt
            loc_txt_dir = f"./data/train/{txt_name}"
            file = open(loc_txt_dir,'r')
            label = int(file.read(1))
            file.close()
            # Move image to folder pos or neg or 1 depending on label
            move_to = "./data/train/" if i < val_idx else "./data/val/" if i < test_idx else "./data/test/"
            if label == 1:
                shutil.move(f"./data/train/{img_name}", move_to + "pos/")
            else:
                shutil.move(f"./data/train/{img_name}", move_to + "neg/")
            # Delete text file
            os.remove(loc_txt_dir)
    