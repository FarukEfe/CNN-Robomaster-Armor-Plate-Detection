from Client import Client
import os

if __name__ == "__main__":
    # Add required data file
    if not os.path.exists("./data"):
        os.mkdir("./data")
    if not os.path.exists("./final_model"):
        os.mkdir("./final_model")
    # Wait for user to add their train and test data
    _ = input(
    '''
    'data' and 'final_model' folders created. Please do the following:

    \tMove your 'train' and 'test' folders inside 'data'. If only 'train' is provided, the program will split a part of it to make the 'test' folder. If no file is provided, the program will exit.

    \tMove the the `.keras` file inside 'final_model'. If not done so, the program will exit and not complete.          
    '''
    )
    # Run Client
    client = Client()
    client.get_architecture()
    _ = client.evaluate()