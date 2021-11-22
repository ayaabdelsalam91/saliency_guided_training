






MNIST Regular Training
-----------------------
- For MNIST Experiments run regular training:  ```python train_MNIST.py ```
- To get saliency maps run : ```python plotSaliencyMaps_MNIST.py```
- To get denstiy maps run: ```python plotSaliencyMapsCompare_MNIST.py```
- To get accuracy drop plots run: ```python maskedAcc_MNIST.py``` than ```python plotCompareMaskedAcc.py```

MNIST Interpretable Training
-----------------------------
- For interpretable training add the following flag to the above commands, here 50% of the features are masked during training.
   
   ```--trainingType interpretable --featuresDroped 0.5 --RandomMasking ```



CIFAR
---------------------
- For CIFAR Experiments run:```python train_CIFAR.py```
- To get saliency maps run:```python plotSaliencyMaps_CIFAR.py```
- To get denstiy maps run:```python plotSaliencyMapsCompare_CIFAR.py```



Bird
---------------------

- Download dataset from https://www.kaggle.com/gpiosenka/100-bird-species
- For Bird Experiments run:```python train_BIRD.py```
- To get saliency maps run:```python plotSaliencyMaps_BIRD.py```
- To get denstiy maps run:```python plotSaliencyMapsCompare_BIRD.py```


