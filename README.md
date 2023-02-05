# ML-Collections
I spend my free time training and playing around with different neural network architectures. All the below codes use **PyTorch** as I feel it more flexible. 

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

## Logistic Regression
- Bank Note Autentication [implementation](Logistic-Regression/BankNote-Authentication.ipynb)
    - Have used glorot initialisation
    - **Remarks** - non standardised train data reached higher accuracy quickly, but wasn't steady

## MultiLayer Perceptron
- Classification of  Flowers using iris Dataset - [Implementation](/MultiLayer-Perceptron/Classifying-Iris-flowers.ipynb).
    - Created model by subclassing `nn.Module`.
    - Saving and reloading model architecture and parameters.
- XOR Classification - [Implementation](/MultiLayer-Perceptron/XOR-Classification.ipynb).
    - Created model with `nn.Sequential`
        - Easy to create cascading layered models.
    - Created model with `nn.Module`(reason commented)
        - Helpful for creating complex models which is not strictly Sequential.
    - Created model with `nn.Module` along with a custom layer.  
- Regression model for Fuel Prediction - [Implementation](/MultiLayer-Perceptron/Predicting-fuel-efficiency.ipynb).
    - Prepocessed the data from web
    - Trained a Regression model
- Classification of MNIST Digits - [Implementation](MultiLayer-Perceptron/Classifying-MNIST-digits.ipynb).
    - Used data from torchvision.datasets
- Loss functions  and its different inputs - [Implementation](MultiLayer-Perceptron/Loss-functions.ipynb).
    - Logits and Probabilities as input in,
        - Binary Cross Entropy Loss
        - Cross Entropy Loss 

## Convolutional Neural Network
- Naive implementation of 1-D & 2-D Convolution in numpy - [Implementation](CNN/1D-and-2D-Convolution-naive.ipynb)
    - Based on the mathematical convolution process, not the *Cross Correlation*
- Classification of MNIST Digits using CNN - [Implementation](CNN/MNIST-digit-recognition-using-CNN.ipynb)
    - Accuracy of 99%
- Smile classification using CelebA dataset - [Implementation](CNN/Smile-classification.ipynb)
    - Performed data augmentation pipeline. (Doubt in random retrival of image)
- Eye Glass classification using CelebA dataset - [Implementation](CNN/EyeGlass-classification.ipynb)
    - Played around by tweeking various parameters. 
    - Observations are noted.

## Transfer Learning
- Check on [here](Transfer-learning).