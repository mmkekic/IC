import os
import pytest

import numpy as np

from pytest import warns

from .. database.load_db import DataSiPM

from .. core.system_of_units import *

from . light_tables_c  import LT_SiPM
from . light_tables_c  import LT_PMT
from . s2_waveforms_c  import create_wfs

from hypothesis                import given
from hypothesis                import settings
from hypothesis.strategies     import floats
from hypothesis.strategies     import integers
from hypothesis.extra.numpy    import arrays


@given(xs=arrays(np.float, 10, elements = floats  (min_value = -500*mm , max_value = 500*mm )),
       ys=arrays(np.float, 10, elements = floats  (min_value = -500*mm , max_value = 500*mm )),
       ts=arrays(np.float, 10, elements = floats  (min_value =    2*mus, max_value = 100*mus)),
       ps=arrays(np.int32, 10, elements = integers(min_value =    10   , max_value = 100    )))
def test_create_wfs_sipms_shape(get_dfs, xs, ys, ts, ps):
    datasipm = DataSiPM('new')
    fname, psf_df, psf_conf = get_dfs['psf']
    el_drift_velocity = 2.5 * mm/mus
    sensor_time_bin   = 1   * mus
    buffer_length     = 200 * mus
    lt = LT_SiPM(fname=fname, sipm_database=datasipm)
    n_sensors = len(lt.sensor_ids)
    waveform = create_wfs(xs, ys, ts, ps, lt, el_drift_velocity, sensor_time_bin, buffer_length)
    assert isinstance(waveform, np.ndarray)
    assert waveform.shape ==  (n_sensors, buffer_length//sensor_time_bin)


@given(xs=arrays(np.float, 10, elements = floats  (min_value = -500*mm , max_value = 500*mm )),
       ys=arrays(np.float, 10, elements = floats  (min_value = -500*mm , max_value = 500*mm )),
       ts=arrays(np.float, 10, elements = floats  (min_value =    2*mus, max_value = 100*mus)),
       ps=arrays(np.int32, 10, elements = integers(min_value =    10   , max_value = 100    )))
def test_create_wfs_pmts_shape(get_dfs, xs, ys, ts, ps):
    fname, lt_df, lt_conf = get_dfs['lt']
    el_drift_velocity = 2.5 * mm/mus
    sensor_time_bin   = 100 * ns
    buffer_length     = 200 * mus
    lt = LT_PMT(fname=fname)
    n_sensors = len(lt.sensor_ids)
    waveform = create_wfs(xs, ys, ts, ps, lt, el_drift_velocity, sensor_time_bin, buffer_length)
    assert isinstance(waveform, np.ndarray)
    assert waveform.shape == (n_sensors, buffer_length//sensor_time_bin)
