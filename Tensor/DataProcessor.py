from tensor_imports import *

# Convert image into float matrix
def convert_to_float(image, label):
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    return image, label

class DataProcessor:

    # MARK: Initializer

    def __init__(self):
        self.dataset = None
        self.train, self.cv, self.test = None, None, None
        self.img_w, self.img_h = 256, 256
        self.batch_size = 32

        dataset = image_dataset_from_directory(
            './data/train',
            labels='inferred', # Get your own output labels from .txt files
            label_mode='binary',
            image_size=[self.img_w,self.img_h], # This will change later
            interpolation='nearest', # Method used when resizing the images
            batch_size=self.batch_size, # Training batch for images (should match the batch_size in model.fit)
            shuffle=True # Shuffle data
        )

        # Turn images into float matrices
        AUTOTUNE = tf.data.experimental.AUTOTUNE
        dataset = (dataset.map(convert_to_float).cache().prefetch(AUTOTUNE))
        # Last two batches are used for cross-validation and test
        train, cv, test = dataset.take(len(dataset) - 2), dataset.skip(len(dataset) - 2).take(1), dataset.skip(len(dataset) - 1).take(1)
        # Assign dataset to attributes
        self.dataset = dataset
        self.train, self.cv, self.test = train, cv, test
    
    # MARK: Getters
    def get_all(self) -> tf.data.Dataset:
        return self.dataset
    
    def get_train(self, augment=False) -> tf.data.Dataset: 
        if augment:
            # Make a sequential model to augment data
            augmentations = Sequential([
                RandomBrightness(factor=0.5),
                RandomFlip(mode="horizontal")
            ])
            
            apply = lambda x, y: (augmentations(x, training=True), y)
            res = self.train.map(apply).cache()
            res = self.train.concatenate(res)
            return res

        return self.train

    def get_cv(self) -> tf.data.Dataset: return self.cv
    
    def get_test(self) -> tf.data.Dataset: return self.test
        