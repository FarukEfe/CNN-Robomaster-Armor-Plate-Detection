from imports import *

# Convert image into float matrix
def convert_to_float(image, label):
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    return image, label

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
        # print(len(dataset)) # Debug

        # Last two batches are used for cross-validation and test
        train_, cv_, test_ = dataset.take(len(dataset) - 2), dataset.skip(len(dataset) - 2).take(1), dataset.skip(len(dataset) - 1).take(1)
        # Turn images into float matrices
        AUTOTUNE = tf.data.experimental.AUTOTUNE
        train = (train_.map(convert_to_float).cache().prefetch(AUTOTUNE))
        cv = (cv_.map(convert_to_float).cache().prefetch(AUTOTUNE))
        test = (test_.map(convert_to_float).cache().prefetch(AUTOTUNE))