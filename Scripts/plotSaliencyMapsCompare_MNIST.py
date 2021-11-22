from matplotlib import pyplot as plt
import matplotlib
import pandas as pd
import numpy as np
import seaborn as sns
# matplotlib.use("TkAgg")




trainingTypes=["MNIST_regular1_model","MNIST_interpretablefeaturesDroped_0.5_RandomMasking_1_model"]


trainingTypesNames=["Trad.","Interpret."]
trainingTypesNames2=["Traditional","Sal. Guided"]
sns.set(font_scale = 2)
FileName="MNIST"

start=0
for figure in range (5):

	MNIST= [i  for i  in range (start,start+10)]

	rows = len(MNIST)

	columns = 4

	dpi=100
	width = 3*columns
	height = 9/6 * rows
	fig, ax = plt.subplots(rows,columns, figsize=(width,height ), dpi=dpi)



	path="./outputs/Graphs/"
	Loc_Graph="./outputs/Graphs/"


	imagePath=path+"TestData_"
	n=0
	for j in range(0,rows):

		img = plt.imread(imagePath+str(MNIST[n])+".png")
		ax[j,0].imshow(img)
		ax[j,0].axis("off")
		n+=1

	n=0
	for j in range(0,rows):

		saliencyValues= np.load("outputs/SaliencyValues/Grad_"+trainingTypes[0]+"_"+str(MNIST[0])+".npy")
		saliencyValues=saliencyValues.flatten()
		Data=np.zeros((2,saliencyValues.shape[0]*2),dtype=object)
		for x in range(2):

			imagePath=path+"saliencyValues_Grad_"+trainingTypes[x]+"_"

			img = plt.imread(imagePath+str(MNIST[n])+".png")
			ax[j,x+1].imshow(img)
			ax[j,x+1].axis("off")

			if(j==0):
				ax[j,x+1].set_title(trainingTypesNames2[x], fontsize=30)
				ax[j,x+2].set_title("Distribution", fontsize=30)

			saliencyValues= np.load("outputs/SaliencyValues/Grad_"+trainingTypes[x]+"_"+str(MNIST[n])+".npy")

			saliencyValues=saliencyValues.flatten()
			saliencyValues = np.interp(saliencyValues, (saliencyValues.min(), saliencyValues.max()), (-1, +1))
			

			if(x==0):
				Data[0,:saliencyValues.shape[0]]=saliencyValues
				Data[1,:saliencyValues.shape[0]]=trainingTypesNames[x]
			else:
				Data[0,saliencyValues.shape[0]:]=saliencyValues
				Data[1,saliencyValues.shape[0]:]=trainingTypesNames[x]
		data=np.transpose(Data)
		df = pd.DataFrame(data,columns=[ 'Gradient','Training Method'])
		df["Gradient"]=df["Gradient"].astype('float64')
		# print(df['Training Method'])

		sns.violinplot( x=df['Training Method'], y=df["Gradient"], ax=ax[j,-1])
		ax[j,-1].yaxis.set_label_position("right")
		ax[j,-1].yaxis.tick_right()


		n+=1

	# ax[j,-1].set_xlabel("Saliency Distribution",fontsize=15)
	plt.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0, top=2)
	plt.savefig(Loc_Graph+"SaliencyMapsCompare_"+FileName+"_"+str(figure)+'.png' , bbox_inches = 'tight',
	        pad_inches = 0)

	# plt.show()
	plt.clf()
	plt.close()
	start+=10

