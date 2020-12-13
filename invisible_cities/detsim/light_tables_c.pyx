import numpy  as np
import pandas as pd

cimport cython
cimport numpy as np

from libc.math cimport            sqrt
from libc.math cimport           floor
from libc.math cimport round as cround



from ..         core import system_of_units as units
from  . light_tables import                  read_lt

cdef class LT:
    """Base abstract class to be inherited from for all LightTables classes.
    It needs get_values_ cython method implemented, as well as zbins_ and sensor_ids_ attributes.
    """
    cdef double* get_values_(self, const double x, const double y, const int sensor_id):
        raise NotImplementedError

    @property
    def zbins(self):
        return np.asarray(self.zbins_)
    @property
    def sensor_ids(self):
        return np.asarray(self.sensor_ids_)

    def get_values(self, const double x, const double y, const int sns_id):
        """ This is only for using within python"""
        cdef double* pointer
        pointer = self.get_values_(x, y, sns_id)
        if pointer!=NULL:
            return np.asarray(<np.double_t[:self.zbins_.shape[0]]> pointer)
        else:
            return np.zeros(self.zbins_.shape[0])


def get_el_bins(el_pitch, el_gap):
    return np.arange(el_pitch/2., el_gap, el_pitch).astype(np.double)

cdef class LT_SiPM(LT):
    cdef readonly:
        double [:] snsx
        double [:] snsy
    cdef:
        double [:, ::1] values
        double psf_bin
        double max_zel
        double max_psf
        double max_psf2
        double inv_bin
        double active_r2

    def __init__(self, *, fname, sipm_database, el_gap=None, active_r=None):
        lt_df, config_df, el_gap, active_r = read_lt(fname, 'PSF', el_gap, active_r)
        lt_df.set_index('dist_xy', inplace=True)
        self.el_gap    = el_gap
        self.active_r  = active_r
        self.active_r2 = active_r**2

        el_pitch  = float(config_df.loc["pitch_z"].value) * units.mm
        self.zbins_    = get_el_bins(el_pitch, el_gap)
        self.values    = np.array(lt_df.values/len(self.zbins_), order='C', dtype=np.double)
        self.psf_bin   = float(lt_df.index[1]-lt_df.index[0])
        self.inv_bin   = 1./self.psf_bin

        self.snsx        = sipm_database.X.values.astype(np.double)
        self.snsy        = sipm_database.Y.values.astype(np.double)
        self.sensor_ids_ = sipm_database.index.values.astype(np.intc)
        self.max_zel     = el_gap
        self.max_psf     = max(lt_df.index.values)
        self.max_psf2    = self.max_psf**2
        self.nsensors    = len(self.sensor_ids_)


    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    @cython.initializedcheck(False)
    cdef double* get_values_(self, const double x, const double y, const int sns_id):
        cdef:
            double dist
            double aux
            unsigned int psf_bin_id
            double xsipm
            double ysipm
            double tmp_x
            double tmp_y
            double*  values
        if sns_id >= self.nsensors:
            return NULL
        xsipm = self.snsx[sns_id]
        ysipm = self.snsy[sns_id]
        tmp_x = x-xsipm; tmp_y = y-ysipm
        dist = tmp_x*tmp_x + tmp_y*tmp_y
        if dist>self.max_psf2:
            return NULL
        if x*x+y*y>=self.active_r2:
            return NULL
        aux = sqrt(dist)*self.inv_bin
        bin_id = <int> floor(aux)
        values = &self.values[bin_id, 0]
        return values

