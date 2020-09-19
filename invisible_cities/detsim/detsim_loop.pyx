# cython: linetrace=True
# cython: profile=True
cimport psf_functions
from .psf_functions cimport PSF
import numpy as np
import pandas as pd

cimport numpy as np
from libc.math cimport sqrt, round, ceil, floor
cimport cython


@cython.boundscheck(False)
@cython.wraparound(False)
def electron_loop(double [:] xs,
                  double [:] ys,
                  double [:] ts,
                  np.ndarray[unsigned long, ndim=1] phs,
                  PSF PSF,
                  double EL_drift_velocity,
                  double sipm_time_bin,
                  int num_bins):
                  
    cdef:
        long [:] sipms_ids = PSF.get_sipm_ids()  
        int nsipms = sipms_ids.shape[0]
        double [:] zs = PSF.get_z_bins()
        np.ndarray[double, ndim=2] sipmwfs = np.zeros([nsipms, num_bins], dtype=np.float64)
        double[:] psf_factors 
        size_t indx_sipm
        size_t indx_el
        size_t indx_z
        double signal
        double time
        size_t indx_time
        double [:] EL_times_
    #lets create vector of EL_times
    num_zs = np.copy(zs)
    zs_bs = num_zs[1]-num_zs[0]
    EL_times = (num_zs+zs_bs/2.)/EL_drift_velocity
    EL_times_ = EL_times.astype(np.float64)
    
    for indx_sipm in range(sipms_ids.shape[0]):
        for indx_el in range(ts.shape[0]):
            if PSF.is_significant(xs[indx_el], ys[indx_el], sipms_ids[indx_sipm])>0:
                psf_factors = PSF.get_values(xs[indx_el], ys[indx_el], sipms_ids[indx_sipm])
                for indx_z in range(zs.shape[0]):
                    time = ts[indx_el]+EL_times_[indx_z]
                    
                    indx_time = <int> floor(time/sipm_time_bin)
                    if indx_time>=num_bins:
                        continue
                    signal = psf_factors[indx_z] * phs[indx_el]
                    sipmwfs[indx_sipm, indx_time] += signal
    return sipmwfs
