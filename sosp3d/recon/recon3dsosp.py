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

def sosp3d_cgsense(kdata: torch.tensor,
                  ktraj: torch.tensor,
                  smaps: torch.tensor,
                  b0maps: Optional[torch.tensor] = None,
                  xinit: Optional[torch.tensor] = None,
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
