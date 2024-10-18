from tensor_imports import *
from TensorModel import *
# IN THIS FILE, WE TEST ALL OF THE MODELS SAVED IN "best_models" FOLDER (IGNORED ON GITHUB) and RETURN THE BEST MODEL / TRAIN IT ON THE WHOLE DS

class ModelTester:

    models = {}

    best_model = {}

    def __init__(self):
        n = os.listdir("Tensor/best_models")
        for i in range(n):
            tensor = TensorModel()
            tensor.load_model(i)
            self.models[f'neural_net_{i}.keras'] = tensor
    
    
