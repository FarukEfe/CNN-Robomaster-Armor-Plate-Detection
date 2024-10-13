from Model import ModelData as MD

if __name__ == "__main__":
    md = MD.ModelData()
    ex1 = md.get_train(False)
    ex2 = md.get_train(True)

    print(len(ex1),len(ex2))