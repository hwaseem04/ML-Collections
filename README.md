# ML-Collections

## Perceptron
- From Scratch [implementation](/Perceptron/Perceptron.ipynb
) - Using NumPy functions.
- From Scratch [implementation](/Perceptron/Torch-Perceptron.ipynb
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
    