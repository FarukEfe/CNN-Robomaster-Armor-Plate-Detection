import os
import shutil
from math import floor

def split_train_test(test_split: int = 0.2):
    # Return cases
    if test_split >= 1:
        print("Test-split cannot be all of dataset")
        return
    if (os.path.exists(f"./data/test/pos")): return
    # Make test dir
    os.mkdir("./data/test")
    # Make moves list
    moves = []
    for _, _, filenames in os.walk(os.getcwd() + f"./data/train/"):
        # Make test start index
        test_idx = floor(len(filenames) - len(filenames) * 0.2)
        # Split based on index (making sure that labels stay with their images)
        for i in range(0,len(filenames) - 1, 2):
            img_name = filenames[i]
            txt_name = filenames[i+1]
            if i >= test_idx:
                moves.append((f"./data/train/{img_name}", "./data/test"))
                moves.append((f"./data/train/{txt_name}", "./data/test"))
    for mv in moves:
        key, dest = mv[0], mv[1]
        shutil.move(key, dest)

def order_dataset(train: bool = True):
    fdir = "train" if train else "test"    
    if (os.path.exists(f".data/{fdir}/pos") or not os.path.exists(f"./data/{fdir}")):
        return
    # Create directories
    moves: list[tuple[str]] = []
    removes = []
    # Create new folders labelled Pos and Neg
    os.mkdir(f"./data/{fdir}/pos")
    os.mkdir(f"./data/{fdir}/neg")
    # Add files based on label
    for _, _, filenames in os.walk(os.getcwd() + f"./data/{fdir}/"):
        # Iterate through each image in list with its subsequent txt
        for i in range(0,len(filenames) - 1,2):
            img_name = filenames[i]
            txt_name = filenames[i+1] # Get label in txt
            loc_txt_dir = f"./data/{fdir}/{txt_name}"
            file = open(loc_txt_dir,'r')
            label = int(file.read(1))
            file.close()
            # Move image to folder pos or neg or 1 depending on label
            if label == 1:
                moves.append((f"./data/{fdir}/{img_name}", f"./data/{fdir}/pos/"))
            else:
                moves.append((f"./data/{fdir}/{img_name}", f"./data/{fdir}/neg/"))
            removes.append(loc_txt_dir)
            # Delete text file
    # Classify all image files
    for mv in moves:
        key, dest = mv[0], mv[1]
        shutil.move(key,dest)
    # Remove all .txt files
    for rm in removes:
        os.remove(rm)

# Create necessary folders to set up workspace (create essential folders ignored by .gitignore)
def env_setup():
    # Create virtual environment and set up libraries: tensorflow, numpy, 
    # Create data file
    os.mkdir(os.getcwd() + "/data")
    # Create best_models in Tensor
    os.mkdir(os.getcwd() + "/Tensor/best_models")

order_dataset()
order_dataset(train=False)