from yolov8_imports import *

class Model:

    def __init__(self) -> None:

        dataset_yaml = {
            'path': os.getcwd() + '/data',
            'train': 'train',
            'val': 'val',
            'names': ['pos','neg']
        }
    
        with open("./dataset_yaml.yaml","w") as file:
            yaml.dump(dataset_yaml, file)
        
        self.model = None
        self.train_results = None
        print(dataset_yaml['path'])
    
    def load_model(self):
        self.model = YOLO("yolov8s.pt")
        print(os.getcwd())
        self.train_results = self.model.train(data=os.getcwd() + "/dataset_yaml.yaml", epochs=10, imgsz=512)
    
    def display_results(self):
        print(self.train_results)

lol = Model()
lol.load_model()