#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg") 
import Helper
import os

if not os.path.exists("./outputs/Graphs"):
    os.makedirs("./outputs/Graphs")





drop=[0, 10 ,20,30,40,50,60,70,80,90]


Files_1=["MNIST_regular1__trueMask","MNIST_interpretablefeaturesDroped_0.5_RandomMasking_1__trueMask"]
DatasetNames=["MNIST"]
saliencyMethods=["Grad","SG","IG", "DL", "GS","DLS","Random"]
saliencyMethods=["Gradient","SmoothGrad","Integrated Gradient", "DeepLift", "Gradient SHAP","DeepSHAP","Random"]

colors=["k","g","c","m","y","b","r"]
lineStyles=["-",":"]


fig, ax = plt.subplots(nrows=1, ncols=6,sharey=True,figsize=(10, 3))
fig.subplots_adjust(top=0.95)

for f in range(len(Files_1)):
	Data=Helper.load_CSV('./outputs/MaskedAcc/'+Files_1[f]+'.csv')
	for s in range(len(saliencyMethods)-1):
		saliencyMethod=saliencyMethods[s]
		maskedAcc=Data[s,1:]
		random=Data[-1,1:]
		
		if(f==0):
			label_= "Reg."
		else:
			label_="Interp."

		ax[s].plot(drop, maskedAcc ,label=label_, color='k',linestyle=lineStyles[f])
		if(f==0):
			ax[s].plot(drop, random ,label='Random', color='r',linestyle="--")

		
		ax[s].set_title(saliencyMethod,fontsize=10)

		ax[s].set_xlabel("% of features removed",fontsize=7)
	
ax[0].set_ylabel("Model Accuracy")

handles, labels = ax[1].get_legend_handles_labels()

fig.suptitle("Tradtional versus interpretable training model accuracy drop for different saliency methods on MNIST" ,fontsize=11,fontweight ="bold")


handlesToPlot=[handles[0],handles[2],handles[1]]

plt.legend(handles =handlesToPlot)
plt.tight_layout()
plt.savefig('./outputs/Graphs/MNIST_regular_vs_interpretable.png')

plt.show()

