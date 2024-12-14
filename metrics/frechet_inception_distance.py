# ---------------------------------------------------------------
# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
#
# This file has been taken from stylegan2-ada-pytorch.
#
# Source:
# https://github.com/NVlabs/stylegan2-ada-pytorch/tree/main/metrics 
#
# The license for these can be found in license/ directory.
# ---------------------------------------------------------------

"""Frechet Inception Distance (FID) from the paper
"GANs trained by a two time-scale update rule converge to a local Nash
equilibrium". Matches the original implementation by Heusel et al. at
https://github.com/bioinf-jku/TTUR/blob/master/fid.py"""

import numpy as np
import scipy.linalg
from . import metric_utils

#----------------------------------------------------------------------------

def compute_fid(opts, max_real, num_gen, t_max=None):
    # Direct TorchScript translation of http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz
    detector_url = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/inception-2015-12-05.pt'
    detector_kwargs = dict(return_features=True) # Return raw features before the softmax layer.

    mu_real, sigma_real = metric_utils.compute_feature_stats_for_dataset(
        opts=opts, detector_url=detector_url, detector_kwargs=detector_kwargs,
        rel_lo=0, rel_hi=0, capture_mean_cov=True, max_items=max_real).get_mean_cov()

    if t_max is None:
        mu_gen, sigma_gen = metric_utils.compute_feature_stats_for_generator(
            opts=opts, detector_url=detector_url, detector_kwargs=detector_kwargs,
            rel_lo=0, rel_hi=1, capture_mean_cov=True, max_items=num_gen).get_mean_cov()
    else:
        mu_gen, sigma_gen = metric_utils.compute_feature_stats_for_generator_with_data(
            opts=opts, detector_url=detector_url, detector_kwargs=detector_kwargs,
            rel_lo=0, rel_hi=1, t_max=t_max, capture_mean_cov=True, max_items=num_gen).get_mean_cov()

    if opts.rank != 0:
        return float('nan')

    m = np.square(mu_gen - mu_real).sum()
    s, _ = scipy.linalg.sqrtm(np.dot(sigma_gen, sigma_real), disp=False) # pylint: disable=no-member
    fid = np.real(m + np.trace(sigma_gen + sigma_real - s * 2))
    return float(fid)

#----------------------------------------------------------------------------
