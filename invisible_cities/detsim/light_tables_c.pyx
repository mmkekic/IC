"""
This is a module to read the light tables files. Since this depends on the format of the table itself,
the table-specific class should inherit from the base LightTable class, providing:
 - get_values_ method, that depends on the x, y position in the EL plane and the internal sensor_id
   the output of the method is a pointer to array of all the values across EL gap for a given x, y bin
   if there are no values for a given x, y, sensor the result is a NULL pointer

 - zbins_      attribute is an  array of positions corresponding to the EL gap partitions
 - sensor_ids_ attribute is an array of internal sensor ids
 - num_sensors - total number of sensors


LT_SiPM and LT_PMT are designed for currently availabel light tables for sipms and pmts;
should be revisited once the format for the lt is fixed and standardized.

"""


import numpy  as np
import pandas as pd

cimport cython
cimport numpy as np

from libc.math cimport            sqrt
from libc.math cimport           floor
from libc.math cimport round as cround



from ..         core import system_of_units as units
from  . light_tables import                  read_lt

cdef class LightTable:
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
    """
    Returns the array of bins position given the bin distance and the total gap
    """
    return np.arange(el_pitch/2., el_gap, el_pitch).astype(np.double)

cdef class LT_SiPM(LightTable):
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
        self.el_gap_width  = el_gap
        self.active_radius = active_r
        self.active_r2 = active_r**2 # compute this once to speed up the get_values_ calls

        el_pitch  = float(config_df.loc["pitch_z"].value) * units.mm
        self.zbins_    = get_el_bins(el_pitch, el_gap)
        self.values    = np.array(lt_df.values/len(self.zbins_), order='C', dtype=np.double)
        self.psf_bin   = float(lt_df.index[1]-lt_df.index[0]) * units.mm #index of psf is the distance to the sensor in mm
        self.inv_bin   = 1./self.psf_bin # compute this once to speed up the get_values_ calls

        self.snsx        = sipm_database.X.values.astype(np.double)
        self.snsy        = sipm_database.Y.values.astype(np.double)
        self.sensor_ids_ = np.arange(len(sipm_database)).astype(np.intc)
        self.max_zel     = el_gap
        self.max_psf     = max(lt_df.index.values)
        self.max_psf2    = self.max_psf**2
        self.num_sensors = len(self.sensor_ids_)

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
        if sns_id >= self.num_sensors:
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



cdef class LT_PMT(LightTable):
    cdef:
        double [:, :, :, ::1] values
        double max_zel
        double max_psf
        double max_psf2
        double inv_binx
        double inv_biny
        double xmin
        double ymin
        double active_r2

    def __init__(self, *, fname, el_gap=None, active_r=None):
        from scipy.interpolate import griddata
        lt_df, config_df, el_gap, active_r = read_lt(fname, 'LT', el_gap, active_r)
        self.el_gap_width  = el_gap
        self.active_radius = active_r
        self.active_r2 = active_r**2 # compute this once to speed up the get_values_ calls

        sensor = config_df.loc["sensor"].value
        #remove column total from the list of columns
        columns = [col for col in lt_df.columns if ((sensor in col) and ("total" not in col))]
        self.zbins_ = get_el_bins(el_pitch, el_gap)
        el_pitch    = el_gap #hardcoded for this specific table

        self.sensor_ids_ = np.arange(len(columns)).astype(np.intc)
        lenz = len(self.zbins)
        # add dimension for z partitions (1 in case of this table)
        self.values = np.asarray(np.repeat(values_aux, lenz, axis=-1), dtype=np.double, order='C')
        self.xmin = xmin
        self.ymin = ymin
        # calculate inverse to speed up calls of get_values_
        self.inv_binx    = 1./bin_x
        self.inv_biny    = 1./bin_y
        self.num_sensors = len(self.sensor_ids_)
        xtable   = lt_df.x.values
        ytable   = lt_df.y.values
        xmin_, xmax_ = xtable.min(), xtable.max()
        ymin_, ymax_ = ytable.min(), ytable.max()
        bin_x = float(config_df.loc["pitch_x"].value) * units.mm
        bin_y = float(config_df.loc["pitch_y"].value) * units.mm
        # extend min, max to go over the active volume
        xmin, xmax = xmin_-np.ceil((self.active_r-np.abs(xmin_))/bin_x)*bin_x, xmax_+np.ceil((self.active_r-np.abs(xmax_))/bin_x)*bin_x
        ymin, ymax = ymin_-np.ceil((self.active_r-np.abs(ymin_))/bin_y)*bin_y, ymax_+np.ceil((self.active_r-np.abs(ymax_))/bin_y)*bin_y
        #create new centers
        x          = np.arange(xmin, xmax+bin_x/2., bin_x).astype(np.double)
        y          = np.arange(ymin, ymax+bin_y/2., bin_y).astype(np.double)
        xx, yy     = np.meshgrid(x, y)
        values_aux = (np.concatenate([griddata((xtable, ytable), lt_df[column], (yy, xx), method='nearest')[..., None]
                                      for column in columns],axis=-1)[..., None]).astype(np.double)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    @cython.initializedcheck(False)
    cdef double* get_values_(self, const double x, const double y, const int sns_id):
        cdef:
            double*  values
            int xindx_, yindx_
        if (x*x+y*y)>=self.active_r2 :
            return NULL
        if sns_id >= self.num_sensors:
            return NULL
        xindx_ = <int> cround((x-self.xmin)*self.inv_binx)
        yindx_ = <int> cround((y-self.ymin)*self.inv_biny)
        values = &self.values[xindx_, yindx_, sns_id, 0]
        return values
