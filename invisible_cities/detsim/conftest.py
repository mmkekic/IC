import os

import numpy  as np
import pandas as pd
import tables as tb

from pytest import fixture
from pytest import    mark

from .. core               import           system_of_units as units
from .. database           import                   load_db
from .. io      .mcinfo_io import        get_sensor_binning
from .. io      .mcinfo_io import load_mcsensor_response_df
from .. io      .dst_io    import                  load_dst

from .  buffer_functions   import               bin_sensors


@fixture(scope="module")
def mc_waveforms(full_sim_file):
    file_in = full_sim_file
    wfs     = load_mcsensor_response_df(file_in, db_file='new', run_no=-6400)

    sns_bins    = get_sensor_binning(file_in)
    pmt_binwid  = sns_bins.bin_width[sns_bins.index.str.contains( 'Pmt')]
    sipm_binwid = sns_bins.bin_width[sns_bins.index.str.contains('SiPM')]
    return wfs.index.levels[0], pmt_binwid.iloc[0], sipm_binwid.iloc[0], wfs


## !! to-do: generalise for all detector configurations
@fixture(scope="module")
def pmt_ids():
    return load_db.DataPMT('new', 6400).SensorID.values


@fixture(scope="module")
def sipm_ids():
    return load_db.DataSiPM('new', 6400).SensorID.values


@fixture(scope="module")
def binned_waveforms(ICDATADIR):
    binned_file = os.path.join(ICDATADIR, 'binned_simwfs.h5')
    with tb.open_file(binned_file) as h5in:
        pmt_bins  = h5in.root.BINWIDTHS.pmt_binwid .read()
        sipm_bins = h5in.root.BINWIDTHS.sipm_binwid.read()

    pmt_wf  = pd.read_hdf(binned_file,  'pmtwfs')
    sipm_wf = pd.read_hdf(binned_file, 'sipmwfs')
    return pmt_bins, pmt_wf, sipm_bins, sipm_wf

@fixture(scope='session')
def get_dfs(ICDATADIR):
    psffname = os.path.join(ICDATADIR, 'NEXT_NEW.tracking.S2.SiPM.LightTable.h5')
    ltfname  = os.path.join(ICDATADIR, 'NEXT_NEW.energy.S2.PmtR11410.LightTable.h5')
    psf_df   = load_dst(psffname, 'PSF', 'LightTable').set_index('dist_xy')
    lt_df    = load_dst( ltfname,  'LT', 'LightTable').set_index(['x', 'y'])
    psf_conf = load_dst(psffname, 'PSF', 'Config'    ).set_index('parameter')
    lt_conf  = load_dst( ltfname,  'LT', 'Config'    ).set_index('parameter')
    return  {'psf':(psffname, psf_df, psf_conf),
             'lt' : (ltfname, lt_df, lt_conf)}

