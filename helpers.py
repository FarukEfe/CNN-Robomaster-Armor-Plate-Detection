import os
import shutil

def order_train():
    # Create new folders labelled Pos and Neg
    os.mkdir("./train/pos")
    os.mkdir("./train/neg")
    # Add files based on label
    for dirname, _, filenames in os.walk(os.getcwd() + "./train/"):
        # Iterate through each image in list with its subsequent txt
        for i in range(0,len(filenames) - 1,2):
            img_name = filenames[i]
            txt_name = filenames[i+1] # Get label in txt
            loc_txt_dir = f"./train/{txt_name}"
            file = open(loc_txt_dir,'r')
            label = int(file.read(1))
            file.close()
            # Move image to folder pos or neg or 1 depending on label
            if label == 1:
                shutil.move(f"./train/{img_name}","./train/pos/")
            else:
                shutil.move(f"./train/{img_name}","./train/neg/")
            # Delete text file
            os.remove(loc_txt_dir)