






MNIST
---------------------
- For MNIST Experiments run: ```python train_MNIST.py --trainingType interpretable --featuresDroped 0.5 --RandomMasking```
- To get saliency maps run: ```python plotSaliencyMaps_MNIST.py --trainingType interpretable --featuresDroped 0.5 --RandomMasking```
- To get denstiy maps run: ```python plotSaliencyMapsCompare_MNIST.py```
- To get accuracy drop plots run: ```python maskedAcc_MNIST.py``` than ```python plotCompareMaskedAcc.py```



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


