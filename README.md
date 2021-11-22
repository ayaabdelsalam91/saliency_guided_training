# Saliency Guided Training
Code implementing "Improving Deep Learning Interpretability by Saliency Guided Training" by Aya Abdelsalam Ismail, Hector Corrada Bravo*, Soheil Feizi*.



![alt text](results.png)

## Overview

Saliency methods have been widely used to highlight important input features in model predictions. Most existing methods use backpropagation on a modified gradient function to generate saliency maps. Thus, noisy gradients can result in unfaithful feature attributions. In this paper, we tackle this issue and introduce a saliency guided training† procedure for neural networks to reduce noisy gradients used in predictions while retaining the predictive performance of the model. Our saliency guided training procedure iteratively masks features with small and potentially noisy gradients while maximizing the similarity of model outputs for both
masked and unmasked inputs. We apply the saliency guided training procedure to various synthetic and real data sets from computer vision, natural language processing, and time series across diverse neural architectures, including Recurrent Neural Networks, Convolutional Networks, and Transformers. Through qualitative and quantitative evaluations, we show that saliency guided training procedure significantly improves model interpretability across various domains while preserving its predictive performance.



 ## Usage:
- Create the following folder structure.
```
Scripts
    │
    ├── data
    ├── models
    └── outputs 
      ├── SaliencyValues
      ├── MaskedAcc
      └──Graphs
```
- To run experiment please refer to README.md in the Scripts folder
