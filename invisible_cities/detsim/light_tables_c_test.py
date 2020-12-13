import os
import pytest

import numpy as np

from pytest import warns

from .. database.load_db    import DataSiPM
from .. io      .dst_io     import load_dst
from .. core.core_functions import find_nearest

from . light_tables_c import LT_SiPM
from hypothesis                import given
from hypothesis                import settings
from hypothesis.strategies     import floats
from hypothesis.strategies     import integers


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

@given(xs=floats(min_value=-500, max_value=500),
       ys=floats(min_value=-500, max_value=500),
       sipm_indx=integers(min_value=0, max_value=1500))
def test_LT_SiPM_values(get_dfs, xs, ys, sipm_indx):
    datasipm = DataSiPM('new')
    fname, psf_df, psf_conf = get_dfs['psf']

    r_active = psf_conf.loc['ACTIVE_rad'].astype(float).value
    r = np.sqrt(xs**2 + ys**2)
    psfbins = psf_df.index.values
    lenz = psf_df.shape[1]
    psf_df = psf_df /lenz
    lt = LT_SiPM(fname=fname, sipm_database=datasipm)
    x_sipm, y_sipm = datasipm.iloc[sipm_indx][['X', 'Y']]
    dist = np.sqrt((xs-x_sipm)**2+(ys-y_sipm)**2)
    psf_bin = np.digitize(dist, psfbins)-1
    max_psf = psf_df.index.max()
    if (dist>=max_psf) or (r>=r_active):
        values = np.zeros(psf_df.shape[1])
    else:
        values = (psf_df.loc[psf_bin].values)

    ltvals = lt.get_values(xs, ys, sipm_indx)
    np.testing.assert_allclose(values, ltvals)

