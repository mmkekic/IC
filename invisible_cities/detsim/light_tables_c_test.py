import os
import pytest

import numpy as np

from pytest import warns

from .. database.load_db    import DataSiPM
from .. io      .dst_io     import load_dst
from .. core.core_functions import find_nearest

from . light_tables_c import LT_SiPM


def test_LT_SiPM_optional_arguments(get_dfs):
    datasipm = DataSiPM('new')
    fname, psf_df, psf_conf = get_dfs['psf']
    lt = LT_SiPM(fname=fname, sipm_database=datasipm)
    #check the values are read from the table
    assert lt.el_gap   == psf_conf.loc['EL_GAP'    ].astype(float).value
    assert lt.active_r == psf_conf.loc['ACTIVE_rad'].astype(float).value
    #check optional arguments are set with User Warning
    with warns(UserWarning):
        lt = LT_SiPM(fname=fname, sipm_database=datasipm, el_gap=2, active_r=150)
        assert lt.el_gap   == 2
        assert lt.active_r == 150

