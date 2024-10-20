from Tensor.tensor_imports import *
from helpers import *
from Tensor.TensorModel import TensorModel

class Client:

    def __init__(self):
        # If training data doesn't exist, exit with code 0
        if (not os.path.exists("./data/train") or len(os.listdir("./final_model")) == 0):
            print("Program couldn't find necessary files, exiting ...")
            exit(0)
        # If test set doesn't exist, split from training data
        if (not os.path.exists("./data/test")):
            split_train_test()
        # Make sure test and train are processed
        order_dataset()
        order_dataset(train=False)
        # Fetch best model selection as a TensorModel into client
        self.tensor = TensorModel()
        self.tensor.load_model(final_model=True)

    def get_architecture(self):
        # Display model architecture
        self.tensor.model.get_build_config() # I don't know the method needed, this is for testing

    def evaluate(self):
        # Evaluate the model on the images in test folder
        result = self.tensor.test_model()
        print(result)