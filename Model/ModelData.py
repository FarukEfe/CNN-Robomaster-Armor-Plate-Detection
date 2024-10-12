from imports import *

class ModelData:

    def __init__(self):
        dataset = image_dataset_from_directory(
            './train',
            labels='inferred', # Get your own output labels from .txt files
            label_mode='binary',
            image_size=[256,256], # This will change later
            interpolation='nearest', # Method used when resizing the images
            batch_size=32, # Training batch for images (should match the batch_size in model.fit)
            shuffle=True # Shuffle data
        )
        print(len(dataset))

        AUTOTUNE = tf.data.experimental.AUTOTUNE
        # train_X, val_X, train_y, val_y
