cdef class LT:
    cdef readonly:
        double el_gap
        double active_r
        int    nsensors
    cdef:
        int    [:] sensor_ids_
        double [:] zbins_
    cdef double* get_values_(self, const double x, const double y, const int sensor_id)
