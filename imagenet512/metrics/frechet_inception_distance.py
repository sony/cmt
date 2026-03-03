# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Frechet Inception Distance (FID) from the paper
"GANs trained by a two time-scale update rule converge to a local Nash
equilibrium". Matches the original implementation by Heusel et al. at
https://github.com/bioinf-jku/TTUR/blob/master/fid.py"""

import numpy as np
import scipy.linalg
from . import metric_utils
import dnnlib
import pickle

#----------------------------------------------------------------------------

def load_stats(path, verbose=True):
    if verbose:
        print(f'Loading feature statistics from {path} ...')
    with dnnlib.util.open_url(path, verbose=verbose) as f:
        if path.lower().endswith('.npz'): # backwards compatibility with https://github.com/NVlabs/edm
            return {'fid': dict(np.load(f))}
        return pickle.load(f)

def compute_fid(opts, max_real, num_gen):
    # Direct TorchScript translation of http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz
    detector_url = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/inception-2015-12-05.pt'
    detector_kwargs = dict(return_features=True) # Return raw features before the softmax layer.

    # mu_real, sigma_real = metric_utils.compute_feature_stats_for_dataset(
    #     opts=opts, detector_url=detector_url, detector_kwargs=detector_kwargs,
    #     rel_lo=0, rel_hi=0, capture_mean_cov=True, max_items=max_real).get_mean_cov()

    ref_url = "https://nvlabs-fi-cdn.nvidia.com/edm2/dataset-refs/img512.pkl"
    ref = load_stats(ref_url, verbose=False)

    mu_real, sigma_real = ref['fid']['mu'], ref['fid']['sigma']

    mu_gen, sigma_gen = metric_utils.compute_feature_stats_for_generator(
        opts=opts, detector_url=detector_url, detector_kwargs=detector_kwargs,
        rel_lo=0, rel_hi=1, capture_mean_cov=True, max_items=num_gen).get_mean_cov()

    if opts.rank != 0:
        return float('nan')

    m = np.square(mu_gen - mu_real).sum()
    s, _ = scipy.linalg.sqrtm(np.dot(sigma_gen, sigma_real), disp=False) # pylint: disable=no-member
    fid = np.real(m + np.trace(sigma_gen + sigma_real - s * 2))
    return float(fid)

#----------------------------------------------------------------------------
