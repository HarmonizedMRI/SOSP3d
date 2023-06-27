"""
Script to reconstruct 3d stack-of-spirals data acquired on 3T GE scanner.
Naveen Murthy (nnmurthy@umich.edu)
2023-04
"""

import torch
import time
import sys
import os
import warnings

# add path to sosp3d for relative imports (todo: better fix?)
[sys.path.append(f) for f in ['.', '..']]
from sosp3d.recon import sosp3d_cgsense, setup_recondata
from sosp3d.utils import im

# paths (to be modified by user)
data_dir = '/n/ir71/d2/nnmurthy/data/20221013_UM3TUHP_3dspiral/out/'
kdata_path = data_dir + "kdata.h5" # k-space data
ktraj_path = data_dir + "ktraj.h5" # spiral trajectories
smaps_path = data_dir + "smaps.h5" # sensitivity maps
b0maps_path = data_dir + "b0maps.h5" # fieldmaps

# save images?
save_images = True
save_folder = '/n/badwater/z/nnmurthy/projects/SOSP3d/figs/' # to be modified by user

# imaging setup
device = torch.device('cuda:0')
N = [92, 92, 42] # image matrix size
res = 0.24 # resolution in cm
FOV = [k*res for k in N] # fov in cm
print("Imaging setup:")
print("N = ", N, "; res = ", res, "; fov = ", FOV)

# setup data for reconstruction
kdata, ktraj, smaps, _ = setup_recondata(kdata_path, ktraj_path, smaps_path, None, device=device)
print("\nkdata.shape:", kdata.shape)
print("\nktraj.shape:", ktraj.shape)
print("\nsmaps.shape:", smaps.shape)

# No off-resonance correction
print("\nRunning iterative recon without B0 correction...")
start = time.time()
xrec_noB0 = sosp3d_cgsense(kdata, ktraj, smaps, b0maps=None, mri_forw_args = {'numpoints': (6,6,1)})
end = time.time()
print("Done.")
print(f"Time taken for iterative recon: {end - start:0.1f} seconds.\n")

# with off-resonance correction
_, _, _, b0maps = setup_recondata(None, None, None, b0maps_path, device=device)
print("\nb0maps.shape:", b0maps.shape)
print("\nRunning iterative recon with off-resonance correction...")
start = time.time()
xrec_B0 = sosp3d_cgsense(kdata, ktraj, smaps, b0maps=b0maps, mri_forw_args = {'numpoints': (6,6,1), 'L': 36})
end = time.time()
print("Done.")
print(f"Time taken for iterative recon: {end - start:0.1f} seconds.\n")

# save images
if save_images:

    # create directory if necessary
    if not os.path.isdir(save_folder):
        warnings.warn("No folder to save images. Creating a directory at {} to save figures.".format(os.path.abspath(save_folder)))
        os.makedirs(save_folder)

    print("Saving images...")
    xrec_comb = torch.cat([xrec_noB0, xrec_B0], dim=2)

    # plot and save recon images for frame 1
    im(torch.abs(xrec_comb[0, ...,12:-1:4]).squeeze().cpu().numpy(), (2,3), transpose=True, savepath = save_folder + 'xrec.png')

    # plot and save fieldmaps
    im(b0maps[...,12:-1:4].squeeze().cpu().numpy(), (2,3), transpose=True, savepath=save_folder + 'b0maps.png', cbar=True, cmap='viridis')
    print("Done.\n")
