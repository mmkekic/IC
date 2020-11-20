import numpy  as np
import tables as tb
import pandas as pd

from functools import partial

from invisible_cities.core import system_of_units as units
from invisible_cities.reco import tbl_functions   as tbl

import invisible_cities.database.load_db          as db

from . components import city
from . components import print_every
from . components import copy_mc_info
from . components import collect

from invisible_cities.dataflow  import dataflow   as fl

from invisible_cities.io.rwf_io           import rwf_writer
from invisible_cities.io.run_and_event_io import run_and_event_writer

# DETSIM IMPORTS
from .  components                import MC_hits_from_files
from .. detsim.simulate_electrons import generate_ionization_electrons
from .. detsim.simulate_electrons import drift_electrons
from .. detsim.simulate_electrons import diffuse_electrons
from .. detsim.lighttables        import LT_SiPM
from .. detsim.lighttables        import LT_PMT
from .. detsim.ielectrons_loop    import electron_loop
from .. detsim.simulate_S1        import create_lighttable_function
from .. detsim.simulate_S1        import compute_S1_pes_at_pmts
from .. detsim.simulate_S1        import generate_S1_times_from_pes
from .. detsim.simulate_S1        import create_S1_waveforms


# @profile
@city
def detsim(files_in, file_out, event_range, detector_db, run_number, s1_lighttable, s2_lighttable, sipm_psf,
           ws, wi, fano_factor, drift_velocity, lifetime, transverse_diffusion, longitudinal_diffusion,
           el_gain, conde_policarpo_factor, drift_velocity_EL,
           wf_buffer_length, wf_pmt_bin_width, wf_sipm_bin_width,
           print_mod, compression):

    ########################
    ######## Globals #######
    ########################
    datapmt  = db.DataPMT (detector_db, run_number)
    datasipm = db.DataSiPM(detector_db, run_number)
    npmts  = len(datapmt)
    nsipms = len(datasipm)

    S1_LT = create_lighttable_function(s1_lighttable)
    S2_LT = LT_PMT (fname=s2_lighttable)
    PSF   = LT_SiPM(fname=sipm_psf, sipm_database=datasipm)

    el_gain_sigma = np.sqrt(el_gain * conde_policarpo_factor)

    # functions
    simulate_ielectrons = electron_simulation(wi, fano_factor, lifetime, transverse_diffusion, longitudinal_diffusion, drift_velocity)
    compute_S1_times    = S1_times_simulation(S1_LT)
    generate_S1_photons = lambda energies: np.random.poisson(energies / ws)
    generate_S2_photons = lambda nes:      np.random.normal (el_gain, el_gain_sigma, size=nes).astype(np.int32)

    create_S2_pmtwfs  = create_S2_waveforms(S2_LT, drift_velocity_EL, wf_pmt_bin_width)
    create_S2_sipmwfs = create_S2_waveforms(PSF  , drift_velocity_EL, wf_sipm_bin_width)

    ##########################################
    ############ SIMULATE ELECTRONS ##########
    ##########################################
    select_active_hits  = fl.map(select_active_hits_, args=("x", "y", "z", "energy", "time", "label"), out=("x_a", "y_a", "z_a", "energy_a", "time_a"))
    # TO-DO: FILTER EMPTY ACTIVE HITS
    simulate_ielectrons = fl.map(simulate_ielectrons, args=("x_a", "y_a", "z_a", "energy_a")         , out=("dx", "dy", "dz", "electrons"))
    count_electrons     = fl.map(lambda x: np.sum(x), args=("electrons"), out=("nes"))
    el_arrival_times    = fl.map(lambda dz, times, electrons: dz/drift_velocity + np.repeat(times, electrons), args=("dz", "time_a", "electrons"), out=("dt"))

    simulate_electrons = fl.pipe(select_active_hits, simulate_ielectrons, count_electrons, el_arrival_times)

    ############################################
    ############ SIMULATE PHOTONS ##############
    ############################################
    generate_S1_photons = fl.map(generate_S1_photons, args = ("energy"), out = ("S1photons"))
    generate_S2_photons = fl.map(generate_S2_photons, args = ("nes")   , out = ("S2photons"))

    simulate_photons = fl.pipe(generate_S1_photons, generate_S2_photons)

    #################################
    ############ TIMES ##############
    #################################
    compute_S1_times      = fl.map(compute_S1_times, args=("x", "y", "z", "time", "S1photons"), out="S1times")
    compute_bias_time     = fl.map(lambda S1times: np.min(np.hstack(S1times)), args=("S1times"), out=("bias_time"))
    translate_S1_times    = fl.map(lambda S1times, bias_time: [times-bias_time for times in S1times], args=("S1times", "bias_time"), out=("S1times"))
    translate_S2_times    = fl.map(lambda S2times, bias_time: S2times-bias_time, args=("dt", "bias_time"), out=("dt"))
    compute_buffer_length = fl.map(lambda S2times: np.ceil(np.max(S2times) + np.max((wf_pmt_bin_width, wf_sipm_bin_width))), args=("dt"), out=("buffer_length"))

    #############################################
    ############ CREATE WAVEFORMS ###############
    #############################################
    ##### PMTs #####
    create_S1_pmtwfs = partial(create_S1_waveforms, bin_width=wf_pmt_bin_width)
    create_S1_pmtwfs = fl.map(create_S1pmtwfs, args=("S1times", "buffer_length"), out=("S1pmtwfs"))
    create_S2_pmtwfs = fl.map(create_S2_pmtwfs, args=("dx", "dy", "dt", "S2photons", "buffer_length"), out=("S2pmtwfs"))
    add_pmtwfs       = fl.map(lambda x, y: x + y, args=("S1pmtwfs", "S2pmtwfs"), out=("pmtwfs"))

    create_pmt_waveforms = fl.pipe(create_S1_pmtwfs, create_S2_pmtwfs, add_pmtwfs)

    #### SIPMs #####
    create_sipm_waveforms = fl.map(create_S2_sipmwfs, args=("dx", "dy", "dt", "S2photons", "buffer_length"), out=("sipmwfs"))

    ############################
    ####### BUFFY PIPE #########
    ############################
    # TO DO


    event_count_in = fl.spy_count()
    evtnum_collect = collect()

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

        result = fl.push(source=MC_hits_from_files(files_in),
                         pipe  = fl.pipe(fl.slice(*event_range, close_all=True),
                                         event_count_in.spy    ,
                                         print_every(print_mod),
                                         simulate_electrons,
                                         simulate_photons,
                                         # fl.spy(lambda d: [print(k) for k in d]),
                                         create_pmt_waveforms,
                                         create_sipm_waveforms,
                                         fl.branch("event_number"     ,
                                                   evtnum_collect.sink),
                                         fl.fork(write_pmtwfs,
                                                 write_sipmwfs,
                                                 write_run_event)),
                         result = dict(events_in     = event_count_in.future,
                                       evtnum_list   = evtnum_collect.future))

        if run_number <= 0:
            copy_mc_info(files_in, h5out, result.evtnum_list,
                         detector_db, run_number)




########################
###### FUNCTIONS #######
########################

def select_active_hits_(x, y, z, energy, time, label):
    sel = label == "ACTIVE"
    return x[sel], y[sel], z[sel], energy[sel], time[sel]


def electron_simulation(wi, fano_factor, lifetime, transverse_diffusion, longitudinal_diffusion, drift_velocity):

    def simulate_ielectrons(x, y, z, energy):
        electrons = generate_ionization_electrons(energy, wi, fano_factor)
        electrons = drift_electrons(z, electrons, lifetime, drift_velocity)
        dx, dy, dz = diffuse_electrons(x, y, z, electrons, transverse_diffusion, longitudinal_diffusion)
        return dx, dy, dz, electrons

    return simulate_ielectrons


def S1_times_simulation(S1_LT):

    def compute_S1_times(x, y, z, time, S1photons):
        S1_pes_at_pmts = compute_S1_pes_at_pmts(x, y, z, S1photons, S1_LT)
        S1times = generate_S1_times_from_pes(S1_pes_at_pmts, time)
        return S1times

    return compute_S1_times


def create_S2_waveforms(LT, EL_drift_velocity, sensor_time_bin):

    def create_waveforms(xs, ys, ts, phs, buffer_length):
        return electron_loop(xs, ys, ts, phs, LT, EL_drift_velocity, sensor_time_bin, buffer_length)

    return create_waveforms
