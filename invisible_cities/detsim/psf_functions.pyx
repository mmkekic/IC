import numpy as np
import pandas as pd
from ..core import system_of_units as units

cimport numpy as np
from libc.math cimport sqrt, round, ceil, floor
cimport cython
cimport cython

#base class for PSF
cdef class PSF:
    cdef double [:] get_values(self, double x, double y, int sipm_id):
        raise NotImplementedError()
    cdef double[:] get_z_bins(self):
        raise NotImplementedError()
    def get_sipm_ids(self):
        raise NotImplementedError()
    cdef int is_significant(self, double x, double y, int sipm_id):
        raise NotImplementedError()
cdef class PSF_distance(PSF):
    cdef public:
        np.float64_t [:, :] psf_values
        np.float64_t [:] xsipms, ysipms, z_bins, sipm_values
        np.int64_t [:] sipm_ids, z_bins_indcs
        double psf_bin, dz_bin, max_zel, max_psf, EL_z
        int org_part

    def __init__(self, sipm_database, psf_fname, z_bins=None):
        PSF      = pd.read_hdf(psf_fname, "/LightTable")
        Config   = pd.read_hdf(psf_fname, "/Config")
        EL_z     = float(float(Config.loc["EL_GAP"].value) * units.mm)
                
        self.EL_z = EL_z
        el_pitch = float(Config.loc["pitch_z"].value) * units.mm

        if z_bins:
           raise NotImplementedError()
        else:
            self.org_part = 1
            self.z_bins = np.arange(0, EL_z+np.finfo(float).eps, el_pitch)
            self.z_bins_indcs = np.arange(len(self.z_bins)) 

        self.sipm_values = np.zeros(len(self.z_bins), dtype=np.float64)
        self.psf_values = PSF.values/len(self.z_bins)
        self.psf_bin    = float(PSF.index[1]-PSF.index[0])
        self.dz_bin = el_pitch

        self.xsipms = sipm_database.X.values.astype(np.float64)
        self.ysipms = sipm_database.Y.values.astype(np.float64)
        self.sipm_ids = sipm_database.index.values.astype(int)
        self.max_zel = EL_z
        self.max_psf = max(PSF.index.values)
        cdef size_t dummy_indx
       
           
    cdef double [:] get_z_bins(self):
        return np.asarray(self.z_bins)

    def get_sipm_ids(self):
        return np.asarray(self.sipm_ids)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef int is_significant(self, double x, double y, int sipm_id):
        cdef double dist = sqrt((x-self.xsipms[sipm_id])*(x-self.xsipms[sipm_id])+(y-self.ysipms[sipm_id])*(y-self.ysipms[sipm_id]))
        if dist>self.max_psf:
            return 0
        else:
            return 1

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef np.float64_t[:] get_values(self, double x, double y, int sipm_id):
        cdef:
            size_t el_indx
            double dist, el_z
            size_t z_indx
            size_t psf_bin
        dist = sqrt((x-self.xsipms[sipm_id])*(x-self.xsipms[sipm_id])+(y-self.ysipms[sipm_id])*(y-self.ysipms[sipm_id]))
        if dist>self.max_psf:
            return self.sipm_values
        #if (self.org_part == 1):
        psf_bin = <size_t> floor(dist/self.psf_bin)
        return  self.psf_values[psf_bin]

