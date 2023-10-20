"""
Reconstruction routines for 3d stack-of-spirals data.
Naveen Murthy, University of Michigan (nnmurthy@umich.edu)
"""

import torch
import h5py
from typing import Optional
from collections import namedtuple
from einops import rearrange
from mirtorch.linear import Gmri, GmriGram, Diff3dgram, NuSense, NuSenseGram
from mirtorch.alg.cg import CG
import torchkbnufft as tkbn

def sosp3d_cgsense(kdata: torch.Tensor,
                  ktraj: torch.Tensor,
                  smaps: torch.Tensor,
                  b0maps: Optional[torch.Tensor] = None,
                  xinit: Optional[torch.Tensor] = None,
                  mri_forw_args = {'numpoints': (6,6,1)},
                  reg_param = 0.0001,
                  CG_args = {'max_iter': 10, 'alert': True},
                  use_Toeplitz_embedding: bool = True
                  ):
    """
    Reconstruct 3d stack-of-spirals MRI data using CG-SENSE algorithm.
    (todo: Should I provide an option to turn off Toeplitz embedding?)

    Arguments:
        kdata : torch.tensor of size [(nbatch), ncoil, nshot, nread] representing 
               k-space data for 3d stack-of-spirals acquisitions
               Note: (nbatch) in brackets suggests that it is an optional dimension.
        ktraj : torch.tensor of size [(nbatch), 3, nshot, nread] representing k-space of trajectories
                    for 3d stack-of-spirals (in units of radians)
        smaps : torch.tensor of size [(nbatch), ncoil, nx, ny, nz] containing sensitivity
                maps for parallel imaging
                
    Options:
        b0maps : torch.tensor of size [(nbatch), nx, ny, nz] containing fieldmaps (in Hz) or None (default)
        xinit  : torch.tensor of size [(nbatch), 1, nx, ny, nz] representing the initial image
                 used to initialize the optimization algorithm or None (default)
        mri_forw_args : Python dict containing keyword arguments to be passed in to the MRI forward model 
                    in the  MIRTorch package.
                    These keyword arguments are passed in to Gmri() if off-resonance correction
                    is to be performed and NuSense() otherwise.
                    Refer the MIRTorch (https://github.com/guanhuaw/MIRTorch)
                    package for more information about these arguments (MIRTorch/mirtorch/linear/mri.py).
        reg_param : Regularization parameter for the CG-SENSE problem.
        CG_args : Python dict containing keyword arguments to be passed into the Conjugate Gradient (CG)
                  algorithm. For more information on these keyword arguments, refer the CG class in
                  MIRTorch toolbox (MIRTorch/mirtorch/alg/cg.py) 

    Outputs:
        xrec : final reconstructed image; torch.tensor of size [(nbatch), 1, nx, ny, nz]
    """

    # add batch dimension if necessary (todo: also add a method to check that the dimensions of all tensors match)
    if kdata.ndim == 3:
        kdata = kdata.unsqueeze(0)

    nbatch, ncoil, nshot, nread = kdata.shape
    nx, ny, nz = smaps.shape[-3:]

    if ktraj.ndim == 3:
        ktraj = ktraj.expand(nbatch, * [-1]*(ktraj.ndim)) # the second * creates a list of -1's of size (ktraj.ndim), and
        # the first * unpacks the list and passes them in as separate arguments into expand().

    if smaps.ndim == 4:
        smaps = smaps.expand(nbatch, * [-1]*(smaps.ndim))

    if b0maps is not None and b0maps.ndim == 3:
        b0maps = b0maps.expand(nbatch, * [-1] *(b0maps.ndim))

    if xinit is not None and xinit.ndim == 4:
        xinit = xinit.expand(nbatch, * [-1]*(xinit.ndim))

    if b0maps is None:
        # forward model without off-resonance correction
        # Note: In the MIRTorch toolbox, NuSense() requires arguments with different dimensions than Gmri(); hence,
        # the need for a rearrange() operation
        Gop = NuSense(smaps, rearrange(ktraj, 'batch dim shot read -> batch dim (shot read)'), **mri_forw_args)
        if use_Toeplitz_embedding:
            Gtg = NuSenseGram(smaps, rearrange(ktraj, 'batch dim shot read -> batch dim (shot read)'), **mri_forw_args)
        else:
            Gtg = Gop.H * Gop

        # Initial estimate of x.
        if xinit is None:
            # Compute density compensated adjoint reconstruction as initial estimate.
            dcf = tkbn.calc_density_compensation_function(ktraj=ktraj[0,0:2,0,:].unsqueeze(0), im_size=(nx, ny))
            dcf = dcf.permute(1,0,-1).unsqueeze(1).repeat(1, 1, nshot, 1)
            xinit = Gop.H * rearrange(dcf * kdata, 'batch coil shot read -> batch coil (shot read)')
            
        y = Gop.H * rearrange(kdata, 'batch coil shot read -> batch coil (shot read)')

    else:
        # forward model with fieldmap correction
        Gop = Gmri(smaps=smaps, zmap=-b0maps, traj=ktraj, **mri_forw_args)
        if use_Toeplitz_embedding:
            Gtg = GmriGram(smaps=smaps, zmap=-b0maps, traj=ktraj, **mri_forw_args)
        else:
            Gtg = Gop.H * Gop

        # Initial estimate of x.
        if xinit is None:
            # Compute density compensated adjoint reconstruction as initial estimate.
            dcf = tkbn.calc_density_compensation_function(ktraj=ktraj[0,0:2,0,:].unsqueeze(0), im_size=(nx, ny))
            dcf = dcf.permute(1,0,-1).unsqueeze(1).repeat(1, 1, nshot, 1)
            xinit = Gop.H * (dcf * kdata)

        y = Gop.H * kdata

    # Run CG-SENSE algorithm.
    T = Diff3dgram(Gop.size_in)
    CG_alg = CG(Gtg + reg_param * T, **CG_args)
    xrec = CG_alg.run(xinit, y)

    return xrec

def setup_recondata(kdata: Optional[str],
                    ktraj: Optional[str],
                    smaps: Optional[str],
                    b0maps: Optional[str],
                    device = torch.device('cuda:0')):
    """
    Read in data from the provided filepaths, and set up data for reconstruction.

    Arguments:
        kdata : None or string pointing to the location of a .h5 file containing 2 datasets:
                '/kdata_r' and '/kdata_i' which represent the real and imaginary 
                parts of the k-space data, each of size [(nbatch), ncoil, nshot, nread]
        ktraj : None or string pointing to the location of a .h5 file containing a dataset called
                '/ktraj' that contains the k-space trajectories of size [(nbatch), 3, nshot, nread] 
                (in units of radians)
        smaps : None or string pointing to the location of a .h5 file containing 2 datasets:
                '/smaps_r' and '/smaps_i' which represent the real and imaginary 
                parts of the sensitivity maps, each of size [(nbatch), ncoil, nx, ny, nz]
        b0maps : None or string pointing to the location of a .h5 file containing a dataset
                 called '/b0maps' which contains the fieldmaps (in Hz) of size [(nbatch), nx, ny, nz]

    Options: 
        device : Device on which to create memory for torch.tensors. Default: 'cuda:0', which is 
                 GPU 0.

    Outputs:
        kdata : k-space data; torch.tensor of size [(nbatch), ncoil, nshot, nread] or None
        ktraj : spiral trajectory data; torch.tensor of size [(nbatch), 3, nshot, nread] or None
        smaps : sensitivity maps; torch.tensor of size [(nbatch), ncoil, nx, ny, nz] or None
        b0maps : fieldmaps; torch.tensor of size [(nbatch), nx, ny, nz] or None

    """

    if kdata is not None:
        with h5py.File(kdata, "r") as hf:
            kdata = torch.tensor(hf['kdata_r'][()] + 1j*hf['kdata_i'][()]).to(device=device, dtype=torch.cfloat)
            kdata = kdata.permute(torch.arange(kdata.ndim - 1, -1, -1).tolist()) # reverse dimensions since h5 files are read 
            # into Python in reversed order.

    if ktraj is not None:
        with h5py.File(ktraj, "r") as hf:
            ktraj = torch.tensor(hf['ktraj'][()]).to(device=device, dtype=torch.float)
            ktraj = ktraj.permute(torch.arange(ktraj.ndim - 1, -1, -1).tolist()) # reverse dimensions since h5 files are read 
            # into Python in reversed order.

    if smaps is not None:
        with h5py.File(smaps, "r") as hf:
            smaps = torch.tensor(hf['smaps_r'][()] + 1j*hf['smaps_i'][()]).to(device=device, dtype=torch.cfloat)
            smaps = smaps.permute(torch.arange(smaps.ndim - 1, -1, -1).tolist()) # reverse dimensions since h5 files are read 
            # into Python in reversed order.

    if b0maps is not None:
        with h5py.File(b0maps, "r") as hf:
            b0maps = torch.tensor(hf['b0maps'][()]).to(device=device, dtype=torch.float)
            b0maps = b0maps.permute(torch.arange(b0maps.ndim - 1, -1, -1).tolist()) # reverse dimensions since h5 files are read 
            # into Python in reversed order.

    return kdata, ktraj, smaps, b0maps


def undersample_sosp_data(kdata:torch.Tensor, ktraj:torch.Tensor, R_xy:int, R_z:int = 1, istart:int = 0):
    """
    Undersample 3D stack-of-spirals (sosp) data retrospectively. Currently, only allows in-plane
    undersampling of spiral shots. todo: Incorporate through-plane undersampling (along z).
    Assumes that "fully sampled" k-space sosp data is passed into the function. 

    Arguments:
        kdata : "Fully sampled" k-space data; torch.Tensor of size [(nbatch), ncoil, nkz, nshot, nread]. 
                Here, nkz = number of kz encodes, nshot = number of shots in each kz encode (for 
                fully sampled spiral data)
        ktraj : spiral trajectory data; torch.Tensor of size [(nbatch), 3, nkz, nshot, nread]
        R_xy : acceleration factor in-plane (has to be an integer divisor of nshot)

    Options:
        R_z : acceleration factor through-plane (along kz). To be implemented (default: 1) 
        istart : Index of starting spiral shot. Has to be in range 0, 1, ... R_xy - 1. (default: 0)
                 Vary this to obtain diversity of undersampling between frames or timepoints in a 
                 dynamic sequence.

    Outputs:
        kdata_us : k-space data; torch.Tensor of size [(nbatch), ncoil, nkz_us, nshot_us, nread], where
                (nkz_us = nkz // R_z), and (nshot_us = nshot // R_xy) 
        ktraj_us : spiral trajectory data; torch.Tensor of size [(nbatch), 3, nkz_us, nshot_us, nread]
    """

    # checks
    assert kdata.shape[-3:-1] == ktraj.shape[-3:-1], "Number of kz encodes and spiral shots must the be the same in both kdata and ktraj."
    nkz, nshot = kdata.shape[-3:-1]
    assert 0 <= istart < R_xy, f"Index of starting spiral shot istart has to be in range [0, R_xy - 1]. Here, {istart = } and {R_xy = }."

    if nshot % R_xy != 0:
        raise ValueError(f"Acceleration factor in-plane {R_xy = } has to be an integer divisor of number of shots {nshot = }.")

    if R_z > 1:
        raise ValueError("Acceleration factor of R_z > 1 not currently implemented in the through-plane direction along kz.")

    # keep only required spiral shots
    nkz_us = nkz // R_z
    nshot_us = nshot // R_xy
    kdata_us = torch.zeros(list(kdata.shape[:-3]) + [nkz_us, nshot_us, kdata.shape[-1]], dtype=kdata.dtype, device=kdata.device)
    ktraj_us = torch.zeros(list(ktraj.shape[:-3]) + [nkz_us, nshot_us, ktraj.shape[-1]], dtype=ktraj.dtype, device=ktraj.device)

    istart_arr = torch.arange(R_xy).roll(-istart) # e.g., if R_xy = 4 and istart = 3, then istart_arr = [3, 0, 1, 2]
    for kz_i in range(nkz_us):

        # starting index of spiral shot for current kz encode
        k = istart_arr[kz_i % R_xy]

        # pick out spiral shots for the current kz encode to satisfy acceleration factor R_xy.
        # This makes sure that we don't pick identical spiral shots in every kz encode.
        kdata_us[..., kz_i, :, :] = kdata[..., kz_i, k::R_xy, :]
        ktraj_us[..., kz_i, :, :] = ktraj[..., kz_i, k::R_xy, :]

    return kdata_us, ktraj_us
    