 
## Transfer Learning
### Classification of flower classes (daisy, dandelion, rose, sunflower, tulip) - [Data](Transfer-learning/tiny_FR)
- I have implemented 3 architectures. Two of which are fine-tuned on pre-trained architectures - Inception-V3 and ConvNext and one of which is a vanilla grade architecture.
- Stored best epoch checkpoints for inference.
- Used custom dataset class to retrieve data
    - **Need to properly apply transformations to train and test data**. As of now I **haven't done data augmentation**. Need to do it and check performance.
- Inception-V3 [Implementation](Transfer-learning/Inception-v3.ipynb)
    - F1 score ~ 0.80
- ConvNeXt [Implementation](Transfer-learning/ConvNeXt.ipynb)
     - F1 score ~ 0.82
- Vanilla Grade sequential CNN [Implementation](Transfer-learning/Vanilla-sequential-CNN.ipynb)
    - F1 score ~ 0.52
    - It is very poor, need to iterate on the architecture. Tried **Ray Tune**, but didn't configure it properly. Need to do that.