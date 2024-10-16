from tensor_imports import *
from DataProcessor import DataProcessor

'''
Training parameters:
 1. Learning Rate (Handled by Adam optimizer)
 2. Batch Size (Generally 32 is fine, but give this a try)
 3. Number of epochs (First catch a glimpse at the performance using low epochs, then for the final train, do more epochs to ensure the model's training)
 4. Optimizer (try some other than adam, such as SGD or RMSprop)

Inference hyper-parameteres to optimize:
 1. Layers
 2. Units
 3. Activation
 4. Regularizer

Convolutional hyper-parameters to optimize:
 1. Number of filters
 2. Strides
 3. Padding
 4. Dropout layers (dropout rate)
 5. Weight initialization methods (Xavier, He initialization)
 7. Kernels (maybe use custom kernel this time?) 
'''

class TensorModel:

    # Dataset Accesss
    dataset: DataProcessor = DataProcessor()

    # Convolutional Block Hyper-Parameters
    conv_hps = {
        'filters': 6,
        'kernel': (3,3),
        'dilation': (1,1),
        'pool': (2,2),
        'padding': 'valid'
    }
    
    # Inference Block Hyper-Parameters
    hps = {
        'activation': 'relu',
        'layer_1': 128,
        'layer_2': 128,
        'layer_3': 64,
        'layer_4': 16,
        'l2': 0.01
    }

    # Where best models are saved
    path = os.getcwd() + "/Tensor/best_models"

    # MARK: Initializer

    def __init__(self) -> None:
        # Set logger for debug
        tf.get_logger().setLevel("INFO")
        tf.autograph.set_verbosity(0)
        tf.get_logger().setLevel(logging.ERROR)
        # Set tensor environment
        SEED = 12345
        np.random.seed(12345)
        tf.random.set_seed(12345)
        os.environ["PYTHONHASHSEED"] = str(SEED)
        # Set initial model to None
        self.model = None
    
    # MARK: Manual Model
    
    def build_model(self):
        m = Sequential()
        # Format Train Data
        m.add(InputLayer(shape=[self.dataset.img_w,self.dataset.img_h,3]))
        m.add(BatchNormalization())
        # Convolutional Block One
        m.add(Conv2D(filters=self.conv_hps['filters'],kernel_size=self.conv_hps['kernel'],dilation_rate=self.conv_hps['dilation'],padding=self.conv_hps['padding'],activation='relu'))
        m.add(MaxPool2D(pool_size=self.conv_hps['pool']))
        # Convolutional Block Two
        #m.add(Conv2D(filters=3,kernel_size=(3,3),dilation_rate=(1,1),padding='valid',activation='relu'))
        #m.add(MaxPool2D())
        # Flatten Before Inference
        m.add(Flatten())
        # Inference Block
        m.add(Dense(units=self.hps['layer_1'],activation=self.hps['activation']))
        m.add(Dense(units=self.hps['layer_2'],activation=self.hps['activation']))
        m.add(Dense(units=self.hps['layer_3'],activation=self.hps['activation']))
        m.add(Dense(units=self.hps['layer_4'],activation=self.hps['activation']))
        m.add(Dense(units=1,activation='sigmoid',kernel_regularizer=L2(self.hps['l2'])))
        self.model = m
    
    def train_model(self,augment: bool = False, epochs: int = 50):
        if self.model == None or type(self.model) != Sequential:
            return
        
        self.model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
        history = self.model.fit(self.dataset.get_train(augment=augment),validation_data=self.dataset.get_cv(),batch_size=self.dataset.batch_size,epochs=epochs,verbose=1)
        print(type(history))
        return history

    def test_model(self):
        if self.model == None:
            return
        
        results = self.model.evaluate(self.dataset.get_test(),verbose=1)
        return results
    
    # MARK: Hyper Model

    def __hyper_model(self,hp):
        m = Sequential()
        # Format Train Data
        m.add(InputLayer(shape=[self.dataset.img_w,self.dataset.img_h,3]))
        m.add(BatchNormalization())
        # Convolutional Block One
        m.add(Conv2D(filters=self.conv_hps['filters'],kernel_size=self.conv_hps['kernel'],dilation_rate=self.conv_hps['dilation'],padding=self.conv_hps['padding'],activation='relu'))
        m.add(MaxPool2D(pool_size=self.conv_hps['pool']))
        # Convolutional Block Two
        #m.add(Conv2D(filters=3,kernel_size=(3,3),dilation_rate=(1,1),padding='valid',activation='relu'))
        #m.add(MaxPool2D())
        # Flatten Before Inference
        m.add(Flatten())
        # Inference Block
        # Fine-tuning
        hp_activation = hp.Choice('activation', values=['relu','tanh']) # Activation tuning
        hp_units_1 = hp.Int('layer_1', min_value = 16, max_value = 256, step = 16) # Tuning for each layer units
        hp_units_2 = hp.Int('layer_2', min_value = 16, max_value = 256, step = 16)
        hp_units_3 = hp.Int('layer_3', min_value = 16, max_value = 128, step = 16)
        hp_units_4 = hp.Int('layer_4', min_value = 16, max_value = 128, step = 16)
        hp_reg_rate = hp.Choice('l2', values=[0.0001,0.0005,0.001,0.005,0.01,0.05,0.1])
        m.add(Dense(units=hp_units_1,activation=hp_activation))
        m.add(Dense(units=hp_units_2,activation=hp_activation))
        m.add(Dense(units=hp_units_3,activation=hp_activation))
        m.add(Dense(units=hp_units_4,activation=hp_activation))
        m.add(Dense(units=1,activation='sigmoid',kernel_regularizer=L2(hp_reg_rate)))

        m.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
        return m

    def tune_hyper(self):
        tuner = kt.Hyperband(
                            self.__hyper_model,
                            objective=['val_accuracy'],
                            max_epochs=15,
                            factor=3,
                            directory='./',
                            project_name='tensor_runs'
                        )
        stop_early = EarlyStopping(monitor='val_loss', patience=6) # Observe improvement in val_loss and if it doesn't change for over 6 epochs, stop training
        # Separate features and labels
        train_X, train_y = self.dataset.train_features_labels()
        # Normalize features and labels between 1 and 0
        #train_X = train_X.astype('float32') / 255.0
        #train_y = train_y.astype('float32') / 255.0
        tuner.search(train_X, train_y, epochs=50, validation_split=0.2, callbacks=[stop_early])
        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
        self.hps = best_hps.values

    # MARK: Save Model

    def save_model(self):
        n_files = len(os.listdir(self.path))
        self.model.save(self.path + f"/neural_net_{n_files}.keras")
    
    def load_model(self, n: int):
        n_files = len(os.listdir(self.path))
        if (n < 0 or n >= n_files):
            return
        self.model = load_model(f"neural_net_{n}.keras")


if __name__ == "__main__":
    # Hyper-parameter Optimization Testing
    tm = TensorModel()
    # First model testing
    tm.build_model()
    tm.train_model(epochs=50)
    tm.test_model()
    # Save model
    tm.save_model()
    '''
    # Hyper-parameter tuning
    _ = input()
    tm.tune_hyper()
    # Test out new hyperparameters
    tm.build_model()
    tm.train_model(epochs=100)
    tm.test_model()
    '''