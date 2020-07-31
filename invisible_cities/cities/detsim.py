import numpy  as np
import tables as tb
import pandas as pd

from functools import partial

from invisible_cities.core import system_of_units as units
from invisible_cities.reco import tbl_functions   as tbl

import invisible_cities.database.load_db          as db

from invisible_cities.cities.components import city
from invisible_cities.cities.components import print_every

from invisible_cities.dataflow  import dataflow   as fl

from invisible_cities.io.rwf_io           import rwf_writer
from invisible_cities.io.run_and_event_io import run_and_event_writer

# DETSIM IMPORTS
from invisible_cities.cities.detsim_source             import load_MC
from invisible_cities.cities.detsim_simulate_electrons import generate_ionization_electrons as generate_ionization_electrons_
from invisible_cities.cities.detsim_simulate_electrons import drift_electrons               as drift_electrons_
from invisible_cities.cities.detsim_simulate_electrons import diffuse_electrons             as diffuse_electrons_
# from invisible_cities.cities.detsim_simulate_electrons import voxelize                      as voxelize_

from invisible_cities.cities.detsim_simulate_signal    import pes_at_pmts
from invisible_cities.cities.detsim_simulate_signal    import generate_S1_times_from_pes    as generate_S1_times_from_pes_

from invisible_cities.cities.detsim_waveforms          import create_pmt_waveforms          as create_pmt_waveforms_
from invisible_cities.cities.detsim_waveforms          import create_sipm_waveforms         as create_sipm_waveforms_

from invisible_cities.cities.detsim_get_psf            import get_psf
from invisible_cities.cities.detsim_get_psf            import get_ligthtables


def get_derived_parameters(detector_db, run_number,
                           s1_ligthtable, s2_ligthtable, sipm_psf,
                           el_gain, conde_policarpo_factor, drift_velocity_EL,
                           wf_buffer_length, wf_pmt_bin_width, wf_sipm_bin_width):
    ########################
    ######## Globals #######
    ########################
    datapmt  = db.DataPMT (detector_db, run_number)
    datasipm = db.DataSiPM(detector_db, run_number)

    S1_LT = get_ligthtables(s1_ligthtable, "S1")
    S2_LT = get_ligthtables(s2_ligthtable, "S2")
    PSF, info = get_psf(sipm_psf)
    EL_dz, el_pitch, n_el_partitions, n_time_bins = info

    el_gain_sigma = np.sqrt(el_gain * conde_policarpo_factor)

    EL_dtime      =  EL_dz / drift_velocity_EL
    s2_pmt_nsamples  = np.max((int(EL_dtime // wf_pmt_bin_width ), 1))
    s2_sipm_nsamples = np.max((int(el_pitch // wf_sipm_bin_width), 1))

    return datapmt, datasipm,\
           S1_LT, S2_LT, PSF,\
           el_pitch, n_el_partitions, n_time_bins, el_gain_sigma,\
           s2_pmt_nsamples, s2_sipm_nsamples


@city
def detsim(files_in, file_out, event_range, detector_db, run_number, s1_ligthtable, s2_ligthtable, sipm_psf,
           ws, wi, fano_factor, drift_velocity, lifetime, transverse_diffusion, longitudinal_diffusion,
           el_gain, conde_policarpo_factor, drift_velocity_EL,
           pretrigger, wf_buffer_length, wf_pmt_bin_width, wf_sipm_bin_width,
           print_mod, compression):

    ########################
    ######## Globals #######
    ########################
    datapmt, datasipm,\
    S1_LT, S2_LT, PSF,\
    el_pitch, n_el_partitions, n_time_bins, el_gain_sigma,\
    s2_pmt_nsamples, s2_sipm_nsamples = get_derived_parameters(detector_db, run_number,
                                                               s1_ligthtable, s2_ligthtable, sipm_psf,
                                                               el_gain, conde_policarpo_factor, drift_velocity_EL,
                                                               wf_buffer_length, wf_pmt_bin_width, wf_sipm_bin_width)
    xsipms, ysipms = datasipm["X"].values, datasipm["Y"].values
    nsipms = len(datasipm)
    npmts  = len(datapmt)

    ##########################################
    ############ SIMULATE ELECTRONS ##########
    ##########################################
    generate_ionization_electrons = partial(generate_ionization_electrons_, wi=wi, fano_factor=fano_factor)
    generate_ionization_electrons = fl.map(generate_ionization_electrons, args = ("energy"), out  = ("electrons"))

    drift_electrons = partial(drift_electrons_, lifetime=lifetime, drift_velocity=drift_velocity)
    drift_electrons = fl.map(drift_electrons, args = ("z", "electrons"), out  = ("electrons"))

    count_electrons = fl.map(lambda x: np.sum(x), args=("electrons"), out=("nes"))

    diffuse_electrons = partial(diffuse_electrons_, transverse_diffusion=transverse_diffusion, longitudinal_diffusion=longitudinal_diffusion)
    diffuse_electrons = fl.map(diffuse_electrons, args = ("x",  "y",  "z", "electrons"), out  = ("dx", "dy", "dz"))

    add_emmision_times = fl.map(lambda dz, times, electrons: dz + np.repeat(times, electrons)*drift_velocity, args = ("dz", "times", "electrons"), out = ("dz"))

    simulate_electrons = fl.pipe(generate_ionization_electrons, drift_electrons, count_electrons, diffuse_electrons, add_emmision_times)

    ############################################
    ############ SIMULATE PHOTONS ##############
    ############################################
    generate_S1_photons = lambda energies: np.random.poisson(energies / ws)
    generate_S1_photons = fl.map(generate_S1_photons, args = ("energy"), out  = ("S1photons"))

    generate_S2_photons = lambda nes: np.random.normal(el_gain, el_gain_sigma, size=nes)
    generate_S2_photons = fl.map(generate_S2_photons, args = ("nes"), out = ("S2photons"))

    simulate_photons = fl.pipe(generate_S1_photons, generate_S2_photons)

    #############################################
    ############ CREATE WAVEFORMS ###############
    #############################################

    #### PMTs ####
    # S1
    compute_S1pes_at_pmts = partial(pes_at_pmts, S1_LT)
    compute_S1pes_at_pmts = fl.map(compute_S1pes_at_pmts, args = ("S1photons", "x", "y", "z"), out  = ("S1pes_at_pmts"))
    generate_S1_times_from_pes = fl.map(generate_S1_times_from_pes_, args=("S1pes_at_pmts"), out=("S1times"))
    compute_S1buffertimes = lambda S1times: [pretrigger + times for times in S1times]
    compute_S1buffertimes = fl.map(compute_S1buffertimes, args=("S1times"), out=("S1buffertimes_pmt"))
    create_S1pmtwfs = create_pmt_waveforms_("S1", wf_buffer_length, wf_pmt_bin_width)
    create_S1pmtwfs = fl.map(create_S1pmtwfs, args=("S1buffertimes_pmt"), out=("S1pmtwfs"))

    create_pmt_S1_waveforms = fl.pipe(compute_S1pes_at_pmts, generate_S1_times_from_pes, compute_S1buffertimes, create_S1pmtwfs)

    # S2
    compute_S2pes_at_pmts = partial(pes_at_pmts, S2_LT)
    compute_S2pes_at_pmts = fl.map(compute_S2pes_at_pmts, args = ("S2photons", "dx", "dy"), out  = ("S2pes_at_pmts"))
    compute_S2buffertimes = lambda zs: pretrigger + zs/drift_velocity
    compute_S2buffertimes = fl.map(compute_S2buffertimes, args=("dz"), out=("S2buffertimes"))
    create_S2pmtwfs = create_pmt_waveforms_("S2", wf_buffer_length, wf_pmt_bin_width)
    create_S2pmtwfs = partial(create_S2pmtwfs, nsamples = s2_pmt_nsamples)
    create_S2pmtwfs = fl.map(create_S2pmtwfs, args=("S2buffertimes", "S2pes_at_pmts"), out=("S2pmtwfs"))

    create_pmt_S2_waveforms = fl.pipe(compute_S2pes_at_pmts, compute_S2buffertimes, create_S2pmtwfs)

    add_pmtwfs = fl.map(lambda x, y: x + y, args=("S1pmtwfs", "S2pmtwfs"), out=("pmtwfs"))
    create_pmt_waveforms = fl.pipe(create_pmt_S1_waveforms, create_pmt_S2_waveforms, add_pmtwfs)

    #### SIPMs ####
    create_sipm_waveforms = create_sipm_waveforms_(wf_buffer_length, wf_sipm_bin_width, nsipms, n_time_bins, s2_sipm_nsamples, xsipms, ysipms, PSF)
    create_sipm_waveforms = fl.map(create_sipm_waveforms, args=("S2buffertimes", "S2photons", "dx", "dy"), out=("sipmwfs"))


    with tb.open_file(file_out, "w", filters = tbl.filters(compression)) as h5out:

        ######################################
        ############# WRITE WFS ##############
        ######################################
        write_pmtwfs  = rwf_writer(h5out, group_name = None, table_name = "pmtrd" , n_sensors = npmts , waveform_length = int(wf_buffer_length // wf_pmt_bin_width))
        write_sipmwfs = rwf_writer(h5out, group_name = None, table_name = "sipmrd", n_sensors = nsipms, waveform_length = int(wf_buffer_length // wf_sipm_bin_width))
        write_pmtwfs  = fl.sink(write_pmtwfs , args=("pmtwfs"))
        write_sipmwfs = fl.sink(write_sipmwfs, args=("sipmwfs"))

        write_run_event = partial(run_and_event_writer(h5out), run_number, timestamp=0)
        write_run_event = fl.sink(write_run_event, args=("event_number"))

        return fl.push(source=load_MC(files_in),
                       pipe  = fl.pipe(fl.slice(*event_range, close_all=True),
                                       print_every(print_mod),
                                       simulate_electrons,
                                       simulate_photons,
                                       # fl.spy(lambda d: [print(k) for k in d]),
                                       create_pmt_waveforms,
                                       create_sipm_waveforms,
                                       fl.fork(write_pmtwfs,
                                               write_sipmwfs,
                                               write_run_event)),
                        result = ())
