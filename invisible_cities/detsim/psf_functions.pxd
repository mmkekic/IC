cdef class PSF:
    cdef double[:] get_values(self, double x, double y, int sipm_id)
    cdef double[:] get_z_bins(self)
    cdef int is_significant(self, double x, double y, int sipm_id)