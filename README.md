# ML-Collections

## Perceptron
- From Scratch [implementation](/perceptron/Perceptron.ipynb
) - Using NumPy functions.
- From Scratch [implementation](/perceptron/Torch-Perceptron.ipynb
) - Using PyTorch functions.

## Linear Regression
- Toy Dataset [implementaion](Linear-Regression/Using-torch-operations.ipynb)  - Using PyTorch functions.
    - Haven't utilized `torch.nn` class to create model, i.e hardcoded loss function and layer definition.
- Toy Dataset [implementaion](Linear-Regression/Using-torchnn-functionalities.ipynb)  - Utilized `torch.nn` class
    - Used PyTorch's function for loss function, parameter updation and layer definition.

## MultiLayer Perceptron
- Classification of  Flowers using iris Dataset - [Implementation](/MultiLayer-Perceptron/Classifying-Iris-flowers.ipynb).
    - Created model by subclassing `nn.Module`.
    - Saving and reloading model architecture and parameters.
- XOR Classification - [Implementation](/MultiLayer-Perceptron/XOR-Classification.ipynb).
    - Created model with `nn.Sequential`
        - Easy to create cascading layered models.
    - Created model with `nn.Module`
        - Helpful for creating complex models which is not strictly Sequential.
    - Created model with `nn.Module` along with a custom layer.  
- Regression model for Fuel Prediction - [Implementation](/MultiLayer-Perceptron/Predicting-fuel-efficiency.ipynb).
    - Prepocessed the data from web
    - Trained a Regression model
- Classification of MNIST Digits - [Implementation](MultiLayer-Perceptron/Classifying-MNIST-digits.ipynb).
    - Used data from torchvision.datasets