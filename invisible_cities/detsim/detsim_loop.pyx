import numpy as np
cimport numpy as np
from libc.math cimport sqrt, round, ceil, floor
cimport cython


#@cython.boundscheck(False)
#@cython.wraparound(False)
def electron_loop(np.ndarray[double, ndim=1] dx,
                  np.ndarray[double, ndim=1] dy,
                  np.ndarray[double, ndim=1] times,
                  np.ndarray[unsigned long, ndim=1] photons,
                  np.ndarray[double, ndim=1] xsipms,
                  np.ndarray[double, ndim=1] ysipms,
                  np.ndarray[double, ndim=2] PSF,
                  np.ndarray[double, ndim=1] distance_centers,
                  np.ndarray[double, ndim=1] EL_times,
                  double sipm_time_bin,
                  int len_sipm_time_bins
):
    cdef int nsipms = xsipms.shape[0]
    cdef np.ndarray[double, ndim=2] sipmwfs = np.zeros([nsipms, len_sipm_time_bins], dtype=np.float64)
    cdef int numel = dx.shape[0]
    cdef int npartitions = PSF.shape[1]
    cdef double EL_bin = distance_centers[1]-distance_centers[0]
    cdef double max_dist = max(distance_centers)
    cdef int sipmwf_timeindx
    cdef double ts
    cdef int psf_bin
    cdef double signal
    for sipmindx in range(nsipms):
        for elindx in range(numel):
            dist = sqrt((dx[elindx]-xsipms[sipmindx])**2+(dy[elindx]-ysipms[sipmindx])**2)
            if dist>max_dist:
                continue
            for partindx in range(npartitions):     
                ts = times[elindx] + EL_times[partindx]
                psf_bin = <int> floor(dist/EL_bin)
                signal = PSF[psf_bin, partindx]/npartitions*photons[elindx]
                sipmwf_timeindx = <int> floor(ts/sipm_time_bin)
                sipmwfs[sipmindx,sipmwf_timeindx] += signal
    return sipmwfs
