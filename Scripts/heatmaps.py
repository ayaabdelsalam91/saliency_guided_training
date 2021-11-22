

from matplotlib import pyplot as plt
import numpy as np

Loc_Graph = './outputs/Graphs/'


def plotExampleWise(input, saveLocation):
    

    fig, ax = plt.subplots()

    cmap='gray'

    plt.axis('off')
  
    cax = ax.imshow(input, interpolation='nearest', cmap=cmap)
    # plt.show()

    plt.gca().set_axis_off()
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
                hspace = 0, wspace = 0)
    plt.margins(0,0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())

    plt.savefig(Loc_Graph+saveLocation+'.png' , bbox_inches = 'tight',
        pad_inches = 0)
    plt.close()


def rescale_linear(array, new_min, new_max):
    """Rescale an arrary linearly."""
    minimum, maximum = np.min(array), np.max(array)
    m = (new_max - new_min) / (maximum - minimum)
    b = new_min - m * minimum
    return m * array + b



def plotGrad(input, saveLocation):
    input=rescale_linear(input, -1, 1)
    
    z_min, z_max = -np.abs(input).max(), np.abs(input).max()


    fig, ax = plt.subplots()


  
    cax = ax.imshow(input, interpolation='nearest', cmap='RdBu', vmin=z_min, vmax=z_max)

    # plt.show()
    plt.colorbar(cax, ax=ax)
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
                hspace = 0, wspace = 0)
    plt.margins(0,0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())

    plt.savefig(Loc_Graph+saveLocation+'.png' , bbox_inches = 'tight',
        pad_inches = 0)
    plt.close()


