import os
import shutil
from math import floor

def order_dataset():    
    if (os.path.exists("./train/pos")):
        return
    # Create directories
    moves: list[tuple[str]] = []
    removes = []
    # Create new folders labelled Pos and Neg
    os.mkdir("./data/train/pos")
    os.mkdir("./data/train/neg")
    # Add files based on label
    for _, _, filenames in os.walk(os.getcwd() + "./data/train/"):
        # Iterate through each image in list with its subsequent txt
        for i in range(0,len(filenames) - 1,2):
            img_name = filenames[i]
            txt_name = filenames[i+1] # Get label in txt
            loc_txt_dir = f"./data/train/{txt_name}"
            file = open(loc_txt_dir,'r')
            label = int(file.read(1))
            file.close()
            # Move image to folder pos or neg or 1 depending on label
            if label == 1:
                moves.append((f"./data/train/{img_name}", "./data/train/pos/"))
            else:
                moves.append((f"./data/train/{img_name}", "./data/train/neg/"))
            removes.append(loc_txt_dir)
            # Delete text file
    # Classify all image files
    for mv in moves:
        key = mv[0]
        dest = mv[1]
        shutil.move(key,dest)
    # Remove all .txt files
    for rm in removes:
        os.remove(rm)

order_dataset()