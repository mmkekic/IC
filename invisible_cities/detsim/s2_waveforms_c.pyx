import numpy  as np
cimport cython

cimport light_tables_c
from light_tables_c cimport LightTable as LT

cimport numpy as np
from libc.math cimport ceil
from libc.math cimport floor


cdef double[:] spread_histogram(const double[:] histogram, int nsmear_left, int nsmear_right):
    """Spreads histogram values uniformly nsmear_left bins to the left and nsmear_right to the right"""
    cdef int nsamples = nsmear_left + nsmear_right
    cdef int l = len(histogram)
    cdef double [:] spreaded = np.zeros(l + nsamples-1)
    cdef int h, i, aux
    cdef double v
    for h in range(nsmear_left, l):
        if histogram[h]>0:
            v = histogram[h]/nsamples
            aux = h-nsmear_left
            for i in range(nsamples):
                spreaded[aux + i] += v
    return spreaded[:l]


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def create_wfs(double [:] xs           ,
               double [:] ys           ,
               double [:] ts           ,
               int    [:] phs          ,
               LT         lt           ,
               double     el_dv        ,
               double     sns_time_bin ,
               double     buffer_length,
               double     tmin = 0    ):

    cdef:
        int nsens         = lt.sensor_ids.shape[0]
        double[:] zs      = lt.zbins
        int num_bins      = <int> ceil (buffer_length/sns_time_bin)
        double [:, :] wfs = np.zeros([nsens, num_bins], dtype=np.double)
        double el_gap     = lt.el_gap_width

    #lets create vector of EL_times
    zs_bs         = el_gap/zs.shape[0]
    time_bs_sns   = zs_bs /el_dv/sns_time_bin
    max_time_sns  = el_gap/el_dv/sns_time_bin
    #el_times corresponding to light_table z partitions
    cdef double [:] el_times = np.arange(time_bs_sns/2.,max_time_sns,time_bs_sns).astype(np.double)

    cdef:
        int snsindx, sns_id, ph_p
        double[::1] lt_factors   = np.empty_like(el_times, dtype=np.double)
        double *    lt_factors_p = &lt_factors[0]
        double time, t_p, x_p, y_p, signal

    for pindx in range(ts.shape[0]):
        x_p  = xs[pindx]
        y_p  = ys[pindx]
        ph_p = phs[pindx]
        t_p  = (ts[pindx]-tmin)/sns_time_bin
        for snsindx in range(nsens):
            lt_factors_p = lt.get_values_(x_p, y_p, snsindx)
            if lt_factors_p != NULL:
                for elindx in range(el_times.shape[0]):
                    time  = t_p + el_times[elindx]
                    tindx = <int> floor(time)
                    if tindx >= num_bins:
                        continue
                    signal = lt_factors_p[elindx] * ph_p
                    wfs[snsindx, tindx] += signal

    #smearing factor in case time_bs_sns is larger than sns_time_bin
    #used in S2 simulation on pmts
    cdef int nsmear = <int> ceil(time_bs_sns)
    cdef int nsmear_r, nsmear_l
    if nsmear>1:
        nsmear_l  = <int> (nsmear/2)
        nsmear_r = nsmear - nsmear_l
        for snsindx in range(nsens):
            wfs[snsindx] = spread_histogram(wfs[snsindx], nsmear_l, nsmear_r)

    return np.asarray(wfs)

