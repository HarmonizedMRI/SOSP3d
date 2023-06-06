"""
Visualization and plotting routines.
Naveen Murthy (nnmurthy@umich.edu)
"""

import matplotlib.pyplot as plt
from typing import Optional
import numpy as np

def im(image, subplot_size, cmap='gray', origin='lower', clim=None,\
       fig_size_inches=(32, 24), transpose=False, cbar=False, axis=None, savepath=Optional[str]):
    """
    Create an array of subplots of images.
    """
    
    fig, axs = plt.subplots(*subplot_size)
    if axs.ndim == 1:
        axs = np.reshape(axs, (-1, axs.shape[0]))

    fig.set_size_inches(*fig_size_inches)
    
    sx, sy = subplot_size
    idx = 0
    for i in range(sx):
        for j in range(sy):
            if idx < image.shape[-1]:
                if transpose:
                    im = axs[i,j].imshow(image[..., idx].T, cmap=cmap, origin=origin)
                else:
                    im = axs[i,j].imshow(image[..., idx], cmap=cmap, origin=origin)
                    
                if clim is not None:
                    im.set_clim(*clim)
                    
                if cbar:
                    cb = plt.colorbar(im, ax = axs[i,j])
                    cb.ax.tick_params(labelsize=30)
                    
                if axis is None:
                    axs[i,j].set_xticks([])
                    axs[i,j].set_yticks([])
                    
#                 axs[i,j].set_title("Index {}".format(idx), fontsize = 32)
                fig.tight_layout()
            else:
                break
                
            idx += 1
    
    plt.subplots_adjust(wspace=0, hspace=0)

    if savepath is not None:
        plt.savefig(savepath)