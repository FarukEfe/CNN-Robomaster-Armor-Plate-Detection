from tensor_imports import *
from TensorModel import *
# IN THIS FILE, WE TEST ALL OF THE MODELS SAVED IN "best_models" FOLDER (IGNORED ON GITHUB) and RETURN THE BEST MODEL / TRAIN IT ON THE WHOLE DS

class ModelTester:

    models = {}

    best_model = {
        'model': None,
        'stats': {}
    }

    def __init__(self):
        n = os.listdir("Tensor/best_models")
        for i in range(n):
            tensor = TensorModel()
            tensor.load_model(i)
            self.models[f'neural_net_{i}.keras'] = tensor
    
    # TO-DO: Get the AVG of test results
    def __test_model(self, model_save: str, rep: int):
        model: TensorModel = self.models[model_save]
        res_list = []
        for _ in range(rep):
            results = model.test_random_batch()
            res_list.append(results)
        return res_list

    # TO-DO: Select best resulting model and train it on the full ds
    def cherry_pick(self, rep: int):
        keys = self.models.keys()

        for key in keys:
            res = self.__test_model(key,rep)
    
        best_pick: TensorModel = None
        best_pick.final_train(epochs=125)
        best_pick.save_model(final=True)
    
    def save_best(self):
        # Get best model and its stats
        best_pick: TensorModel = self.best_model['model']
        best_stats: dict = self.best_model['stats']
        # Display best model stats
        pass
        # Save best model
        best_pick.final_train(epochs=125)
        best_pick.save_model(final=True)


