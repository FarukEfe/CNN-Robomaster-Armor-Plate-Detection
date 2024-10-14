from tensor_imports import *
from DataProcessor import DataProcessor

'''
Training parameters:
 1. Learning Rate (Handled by Adam optimizer)
 2. Batch Size (Generally 32 is fine, but give this a try)
 3. Number of epochs (First catch a glimpse at the performance using low epochs, then for the final train, do more epochs to ensure the model's training)
 4. Optimizer (try some other than adam, such as SGD or RMSprop)

Of a Convolutional Neural Network, hyper-parameter optimization is important. Here's a list of hyper-parameters you need to optimize:
 1. Number of filters
 2. Strides
 3. Padding
 4. Dropout layers (dropout rate)
 5. Weight initialization methods (Xavier, He initialization)
 6. Regularization (L1, L2) (usually L2 is more effective)
 7. Kernels (maybe use custom kernel this time?) 
'''

class TensorModel:

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
        self.model_data: DataProcessor = DataProcessor()
    
    def build_model(self):
        m = Sequential()
        # Format Train Data
        m.add(InputLayer(shape=[self.model_data.img_w,self.model_data.img_h,3]))
        m.add(BatchNormalization())
        # Convolutional Block
        m.add(Conv2D(filters=3,kernel_size=(3,3),dilation_rate=(1,1),padding='valid',activation='relu'))
        m.add(MaxPool2D()) # def. pool size is 2x2
        m.add(Flatten())
        # Inference Block
        m.add(Dense(units=128,activation='relu'))
        m.add(Dense(units=128,activation='relu'))
        m.add(Dense(units=64,activation='relu'))
        m.add(Dense(units=16,activation='relu'))
        m.add(Dense(units=1,activation='sigmoid',kernel_regularizer=L2(0.05)))
        self.model = m
    
    def train_model(self,augment: bool = False):
        if self.model == None:
            return
        
        self.model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
        history = self.model.fit(self.model_data.get_train(augment=augment),validation_data=self.model_data.get_cv(),batch_size=self.model_data.batch_size,epochs=50,verbose=1)
        print(type(history))
        return history

    def test_model(self):
        if self.model == None:
            return
        
        results = self.model.evaluate(self.model_data.get_test(),verbose=1)
        return results

    def save_model(self):
        self.model.save("neural_net.keras")
    
    def load_model(self):
        self.model = load_model("neural_net.keras")

'''
model = TensorModel()
model.build_model()
model.train_model( augment = False ) 
model.test_model()     
model.save_model()
'''
tm = TensorModel()
tm.load_model()
tm.test_model()