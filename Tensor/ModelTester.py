from tensor_imports import *
from TensorModel import *
# IN THIS FILE, WE TEST ALL OF THE MODELS SAVED IN "best_models" FOLDER (IGNORED ON GITHUB) and RETURN THE BEST MODEL / TRAIN IT ON THE WHOLE DS

class ModelTester:

    models = {}

    best_model = {
        'model': None,
        'loss': -1,
        'acc': -1
    }

    def __init__(self):
        n = os.listdir("Tensor/best_models")
        for i in n:
            tensor = TensorModel()
            tensor.load_model(i)
            self.models[i] = tensor
    
    def show_models(self):
        keys = self.models.keys()
        print(keys)
    
    def show_best(self):
        print(self.best_model)

    # Get the AVG of test results
    def __test_model(self, model_save: str, rep: int) -> float:
        model: TensorModel = self.models[model_save]
        acc: float = 0
        loss: float = 0
        for _ in range(rep):
            results = model.test_random_batch()
            acc += results[1] # Add test accuracy to total
            loss += results[0] # Add test loss to total
        acc /= rep
        loss /= rep
        return acc, loss # Return average accuracy

    # TO-DO: Select best resulting model and train it on the full ds
    def cherry_pick(self, rep: int):
        keys = self.models.keys()
        best_acc, best_loss, best_key = -1, -1, None
        for key in keys:
            acc, loss = self.__test_model(key,rep)
            print(f"{key} has accuracy {acc} and loss {loss}")
            best_key = key if acc > best_acc else best_key
            best_loss = loss if acc > best_acc else best_loss
            # Ensure best_acc updates last since the above values depend on acc > best_acc comparisons
            best_acc = acc if acc > best_acc else best_acc
        self.best_model['model'] = self.models[key]
        self.best_model['acc'] = best_acc
        self.best_model['loss'] = best_loss
    
    def save_best(self):
        # If no model is selected, return without saving
        if self.best_model['model'] == None:
            print("No model selected, aborting from operation ...")
            return
        # Get best model and its stats
        best_pick: TensorModel = self.best_model['model']
        best_stats: dict = { 'acc': self.best_model['acc'], 'loss': self.best_model['loss'] }
        # Display best model stats
        print(best_stats)
        _ = input("If you want to abando, ctrl + c.\n")
        # Save best model
        best_pick.final_train(epochs=125)
        best_pick.save_model(final=True)

if __name__ == "__main__":
    tester = ModelTester()
    tester.show_models()
    tester.cherry_pick(5)
    tester.show_best()
    #tester.save_best()