"""
Script to reconstruct 3d stack-of-spirals data acquired on 3T GE scanner.
We take a fully-sampled stack-of-spirals dataset, and retrospectively undersample it to get 1 shot per kz-encode.
Reconstruction with and without off-resonance correction are compared.


Naveen Murthy (nnmurthy@umich.edu)
2025-02
"""

import torch
import time
import sys
import os

# paths to be modified by user
mirtorch_path = "/n/projects/test_sosp3d/MIRTorch" # path to MIRTorch
data_path = "/n/projects/SOSP3d/data_fromgdrive/stack-of-spirals-dataset/" # path to stack-of-spirals data from Google drive
save_images = True
save_path = '/n/projects/SOSP3d/figs/' # folder to save images in

# add path to sosp3d for relative imports 
[sys.path.append(f) for f in ['.', '..']]
sys.path.insert(0, mirtorch_path) # path to MIRTorch
from sosp3d.recon import sosp3d_cgsense, setup_recondata, undersample_sosp_data
from sosp3d.utils import im

# imaging setup
device = torch.device('cuda:0') # change to "cpu" if you want to run on cpu
N = [92, 92, 42] # image matrix size
nkz = N[-1] # number of kz encodes
nrot = 3 # number of rotated spiral shots per kz encode
res = 0.24 # resolution in cm
FOV = [k*res for k in N] # fov in cm
print("Imaging setup:")
print("N = ", N, "; res = ", res, "; fov = ", FOV)


# setup data for reconstruction. This corresponds to a fully sampled stack-of-spirals acquisition.
kdata, ktraj, smaps, b0maps = setup_recondata(data_path + "kdata.h5", data_path + "ktraj.h5", data_path + "smaps.h5", data_path + "b0maps.h5", device=device)
_,ncoil,_,nread = kdata.shape
kdata = kdata.reshape(1, ncoil, nkz, nrot, nread) # [1, 32, 126, 3570] -> [1, 32, 42, 3, 3570] corresponding to 42 kz-encodes and 3 shots per kz
ktraj = ktraj.reshape(3, nkz, nrot, nread) # [3, 126, 3570] -> [3, 42, 3, 3570]

# Retrospectively undersample stack-of-spirals data to get 1 rotated spiral per kz-encode.
kdata_us, ktraj_us = undersample_sosp_data(kdata, ktraj, nrot, istart=0)
kdata_us = kdata_us.squeeze(3) # [1, 32, 42, 1, 3570] -> [1, 32, 42, 3570]
ktraj_us = ktraj_us.squeeze(2) # [3, 42, 1, 3570] -> [3, 42, 3570]
print("\nkdata_us.shape:", kdata_us.shape)
print("\nktraj_us.shape:", ktraj_us.shape)
print("\nsmaps.shape:", smaps.shape)

# No off-resonance correction
print("\nRunning iterative recon without B0 correction...")
start = time.time()
xrec_noB0 = sosp3d_cgsense(kdata_us, ktraj_us, smaps, b0maps=None, mri_forw_args = {'numpoints': (6,6,1)})
end = time.time()
print("Done.")
print(f"Time taken for iterative recon: {end - start:0.1f} seconds.\n")

# with off-resonance correction
print("\nb0maps.shape:", b0maps.shape)
print("\nRunning iterative recon with off-resonance correction...")
start = time.time()
xrec_B0 = sosp3d_cgsense(kdata_us, ktraj_us, smaps, b0maps=b0maps, mri_forw_args = {'numpoints': (6,6,1), 'L': 36})
end = time.time()
print("Done.")
print(f"Time taken for iterative recon: {end - start:0.1f} seconds.\n")

# save images
if save_images:

    print("Saving images...")
    xrec_comb = torch.cat([xrec_noB0, xrec_B0], dim=2)

    # plot and save recon images for frame 1
    im(torch.abs(xrec_comb[0, ...,12:-1:4]).squeeze().cpu().numpy(), (2,3), transpose=True, savepath = save_path + 'xrec_combined.png')

    # plot and save fieldmaps
    im(b0maps[...,12:-1:4].squeeze().cpu().numpy(), (2,3), transpose=True, savepath=save_path + 'b0maps.png', cbar=True, cmap='viridis')
    print("Done.\n")
