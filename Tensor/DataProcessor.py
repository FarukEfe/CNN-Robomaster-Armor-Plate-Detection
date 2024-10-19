from tensor_imports import *

AUTOTUNE = tf.data.experimental.AUTOTUNE

# Convert image into float matrix
def convert_to_float(image, label):
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    return image, label

class DataProcessor:

    # MARK: Initializer

    def __init__(self):
        self.train, self.cv, self.test = None, None, None
        self.img_w, self.img_h = 256, 256
        self.batch_size = 32
        # Retrieve train dataset
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
        dataset = (dataset.map(convert_to_float).cache().prefetch(AUTOTUNE))
        # Last two batches are used for cross-validation and test
        train, cv = dataset.take(len(dataset) - 1), dataset.skip(len(dataset) - 1).take(1)
        # Get the test split
        test = image_dataset_from_directory(
            './data/test',
            labels='inferred', # Get your own output labels from .txt files
            label_mode='binary',
            image_size=[self.img_w,self.img_h], # This will change later
            interpolation='nearest', # Method used when resizing the images
            batch_size=self.batch_size, # Training batch for images (should match the batch_size in model.fit)
            shuffle=True # Shuffle data
        )
        # Assign dataset to attributes
        self.train, self.cv, self.test = train, cv, test
    
    # MARK: Getters
    
    def get_train(self, augment=False) -> tf.data.Dataset: 
        if augment:
            # Make a sequential model to augment data
            augmentations = Sequential([
                RandomBrightness(factor=0.5),
                RandomFlip(mode="horizontal")
            ])
            
            apply = lambda x, y: (augmentations(x, training=True), y)
            res = (self.train.map(apply).cache().prefetch(AUTOTUNE))
            res = self.train.concatenate(res)
            return res

        return self.train

    # CONSIDER REMOVING THIS IF YOU WON'T USE IT IN THE FUTURE
    def train_features_labels(self) -> tuple[tf.data.Dataset,tf.data.Dataset]:
        features = []
        labels = []
        for feature, label in self.train:
            features.append(feature)
            labels.append(label)
        return features, labels

    def get_cv(self) -> tf.data.Dataset: return self.cv
    
    def get_test(self) -> tf.data.Dataset: return self.test

    def get_random(self, batches: int) -> tf.data.Dataset:
        if batches <= 0: return []
        if batches > len(self.test): return self.test
        sz_buffer = self.batch_size*len(self.test)
        shuffled = self.test.shuffle(buffer_size=sz_buffer,seed=None)
        sampled = shuffled.take(1)
        return sampled.take(batches)
        