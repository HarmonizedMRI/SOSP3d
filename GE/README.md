This folder contains MATLAB code to obtain k-space data and spiral trajectories
from P-files and .mod files acquired using GE scanners. The following
section details a few toolboxes that need to be downloaded for 
preprocessing data acquired on GE scanners.

## Installation: MATLAB toolboxes

1. TOPPE

For reading and manipulating data (P-files) acquired on GE scanners, download the 
[TOPPE](https://github.com/toppeMRI/toppe)
package. Clone the TOPPE folder locally so that it can be accessed within MATLAB.
  
  
2. BART

To estimate sensitivity maps for parallel imaging, install the 
  [BART](https://mrirecon.github.io/bart/installation.html) toolbox, 
  which is an open-source image recon framework for MRI. Add the local path on
  MATLAB so that BART can be accessed.

3. MIRT

Install the Michigan Image Reconstruction Toolbox ([MIRT](https://web.eecs.umich.edu/~fessler/code/))
and run the command `setup` from the MATLAB commandline.
