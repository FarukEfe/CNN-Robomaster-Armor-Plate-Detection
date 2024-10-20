IMPORTANT !!!
=============
In order to properly run the project and test the model, you first need to follow the steps below to set up the project environment. This step includes:

- Setting up a virtual environment `.env`
- Installing the required libraries (`tensorflow`, `numpy`, etc.)
- Creating the necessary project folders.

### Set Up Virtual Environment
To set up the virtual environment, first go to the command line and navigate to the **project root directory**, which will conveniently be named as the title of this GitHub repo.

Once you're in the project rood directory, type `pip3 install virtualenv` if you do not have virtual environment tools installed in your pip version. 

After, type the following command in the command line tool: `python3 -m venv .env`. This line will create a `.env` folder that will hold all of the subsequent library installs and project settings.

However, first we need to activate the project environment, for which we'll run `.env/Scripts/activate` in the command line, from the project root directory.

###### BEFORE PROCEEDING, KNOW THAT THE ONLY FILE YOU'RE SUPPOSED TO USE IN TESTING THE MODEL IS CLIENT.PY (THIS IS VERY IMPORTANT)

### Run __init__.py

Run this file and follow the instructions to load the data. The instructions will be as follows

### Add Training Data

Once you're running \__init\__.py, the command prompt will ask you to move over your 'train' and 'test' folders before hitting `Enter`. So:

Make sure to add your training & testing folders (named 'train' and 'test') inside the 'data' folder.

Even if you're only planning to test the model, it is required to include the 'train' folder as the program cannot initialize `DataProcessor` (and subsequently `TensorModel`, `ModelTester`, `Client`) classes. So it'll exit. **AVOID GETTING THIS ERROR**

You can find the training folder I've used in this link

### Add Final Model

The `.keras` file exceeds the file size allowance by GitHub. So do the following to access my final model.

To find the `.keras` file of the final model I've made, you can navigate to the following google drive link, download, and add it to the `final_model` folder found in the project repository. Here's the link to google drive:

**HERE COMES THE LINK**

### Run & Test

Now, you're ready to run `Client.py` to test the performance of the model with your own testing set. So go ahead and run `Client.py`.

ABOUT THE MODEL
===============

The model holds two convolutional blocks and an inference block.

### 1st Convolutional Block

Layers:

- `Conv2D`: 6 filters and 3x3 kernel. 6 filters is within reasonable range for the 1st block and 3x3 kernel helps us focus more on the localized features instead of the whole image. **ReLU** activation is used to highlight / maximize features.
- `MaxPool2D`: 2x2 pool size (default). This layer helps us simplify the features in a pool of pixels, from which we can extract more features

### 2nd Convolutional Block

Layers:

- `Conv2D`: 12 filters and 3x3 kernel. From a pooled matrix it's easier to gather more features and computationally less demanding. 3x3 kernel and **ReLU** activation are kept here for the same reasons.
- `MaxPool2D`: default pool size. Serves the same purpose as previous pooling layer. 

### Inference Block

Inference in a neural network usually starts with high neuron numbers to infer outcome from very localized information. Dropout is applied to avoid overfitting on certain examples (a random subset of training examples are dropped in each iteration). The number of neurons in the hidden layers reduce towards the output layer, assembling the localized information into more global outcomes. As a result, the output layer assembles the final few pieces to draw a conclusion.

Layers:
- 1st Hidden Layer:
- Dropout Layer:
- 2nd Hidden Layer: 
- 3rd Hidden Layer:
- Dropout Layer: 
- 4th Hidden Layer: 
- Output Layer: Sigmoid activation is used for binary classification. Only 1 neuron is needed (dimensions would mismatch otherwise).

FOR CONTACT
===========
You can reach out to me on my McMaster email: yencilef@mcmaster.ca

Or my personal email: efefrk30@gmail.com