import numpy  as np
import tables as tb
import pandas as pd
import os
from functools import partial

from .. core import system_of_units as units
from .. reco import tbl_functions   as tbl

from .. database import load_db  as db

from . components import city
from . components import print_every
from . components import copy_mc_info
from . components import collect
from . components import calculate_and_save_buffers
from .. dataflow  import dataflow   as fl

from .. io.rwf_io           import rwf_writer
from .. io.run_and_event_io import run_and_event_writer
from .. io. event_filter_io import event_filter_writer

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
           wf_pmt_bin_width, wf_sipm_bin_width,
           max_time   ,
           buffer_length, pre_trigger, trigger_threshold,
           print_mod, compression):

    ########################
    ######## Globals #######
    ########################
    datapmt  = db.DataPMT (detector_db, run_number)
    datasipm = db.DataSiPM(detector_db, run_number)
    npmt  = len(datapmt)
    nsipm = len(datasipm)
    nsamp_pmt  = int(buffer_length /  wf_pmt_bin_width)
    nsamp_sipm = int(buffer_length /  wf_sipm_bin_width)
    LT_pmt  = LT_PMT (fname=os.path.expandvars(s2_lighttable))
    LT_sipm = LT_SiPM(fname=os.path.expandvars(sipm_psf), sipm_database=datasipm)
    el_gap = LT_sipm.el_gap
    el_gain_sigma = np.sqrt(el_gain * conde_policarpo_factor)


    select_active_hits = fl.map(select_active_hits_,
                                args = ('x', 'y', 'z', 'energy', 'time', 'label'),
                                out = ('x_a', 'y_a', 'z_a', 'energy_a', 'time_a'))

    filter_events_no_active_hits = fl.map (lambda x:np.any(x),
                                           args= 'energy_a',
                                           out = 'passed')
    events_passed_active_hits = fl.count_filter(bool, args='passed')

    simulate_electrons = fl.map(ielectron_simulator(wi, fano_factor, lifetime, transverse_diffusion, longitudinal_diffusion, drift_velocity, el_gain, el_gain_sigma),
                                args = ('x_a', 'y_a', 'z_a', 'time_a', 'energy_a'),
                                out  = ('x_ph', 'y_ph', 'z_ph', 'times_ph', 'nphotons'))

    simulate_S1_times = fl.map(s1_times_simulator(s1_lighttable, ws),
                               args = ('x', 'y', 'z', 'time', 'energy'),
                               out = 's1_times')
    get_buffer_times_and_length = fl.map(buffer_times_and_length_getter(wf_pmt_bin_width, wf_sipm_bin_width, el_gap, drift_velocity_EL, S2tmax=max_time),
                                         args = ('s1_times', 'times_ph'),
                                         out = ('tmin', 'tmax', 'buffer_length'))

    create_pmt_s1_waveforms = fl.map(s1_waveforms_creator(wf_pmt_bin_width),
                                     args = ('s1_times', 'tmin', 'buffer_length'),
                                     out = 's1_pmt_waveforms')
    create_pmt_s2_waveforms = fl.map(s2_waveform_creator (wf_pmt_bin_width, LT_pmt, drift_velocity_EL),
                                     args = ('x_ph', 'y_ph', 'times_ph', 'nphotons', 'tmin', 'buffer_length'),
                                     out = 's2_pmt_waveforms')
    sum_pmt_waveforms = fl.map(lambda x, y : x+y,
                                  args = ('s1_pmt_waveforms', 's2_pmt_waveforms'),
                                  out = 'pmt_bin_wfs')
    create_pmt_waveforms = fl.pipe(create_pmt_s1_waveforms, create_pmt_s2_waveforms, sum_pmt_waveforms)

    create_sipm_waveforms = fl.map(s2_waveform_creator (wf_sipm_bin_width, LT_sipm, drift_velocity_EL),
                                   args = ('x_ph', 'y_ph', 'times_ph', 'nphotons', 'tmin', 'buffer_length'),
                                   out = 'sipm_bin_wfs')

    get_bin_edges  = fl.map(bin_edges_getter(wf_pmt_bin_width, wf_sipm_bin_width),
                            args = ('pmt_bin_wfs', 'sipm_bin_wfs'),
                            out = ('pmt_bins', 'sipm_bins'))


    event_count_in = fl.spy_count()
    evtnum_collect = collect()

    with tb.open_file(file_out, "w", filters = tbl.filters(compression)) as h5out:
        buffer_calculation = calculate_and_save_buffers(buffer_length    ,
                                                        pre_trigger      ,
                                                        wf_pmt_bin_width ,
                                                        wf_sipm_bin_width,
                                                        trigger_threshold,
                                                        h5out            ,
                                                        run_number       ,
                                                        npmt             ,
                                                        nsipm            ,
                                                        nsamp_pmt        ,
                                                        nsamp_sipm       )

        write_nohits_filter = fl.sink( event_filter_writer(h5out, "active_hits")   , args=("event_number", "passed"))
        result = fl.push(source=MC_hits_from_files(files_in),
                         pipe  = fl.pipe(fl.slice(*event_range, close_all=True),
                                         event_count_in.spy    ,
                                         print_every(print_mod),
                                         select_active_hits,
                                         filter_events_no_active_hits,
                                         fl.branch(write_nohits_filter) ,
                                         events_passed_active_hits.filter,
                                         simulate_electrons,
                                         simulate_S1_times,
                                         get_buffer_times_and_length,
                                         # fl.spy(lambda d: [print(k) for k in d]),
                                         create_pmt_waveforms,
                                         create_sipm_waveforms,
                                         get_bin_edges,
                                         fl.branch("event_number"     ,
                                                   evtnum_collect.sink),
                                         buffer_calculation),
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


def ielectron_simulator(wi, fano_factor, lifetime, transverse_diffusion, longitudinal_diffusion, drift_velocity, el_gain, el_gain_sigma):
    def simulate_ielectrons(x, y, z, time, energy):        
        electrons = generate_ionization_electrons(energy, wi, fano_factor)
        electrons = drift_electrons(z, electrons, lifetime, drift_velocity)
        dx, dy, dz = diffuse_electrons(x, y, z, electrons, transverse_diffusion, longitudinal_diffusion)
        dtimes  = dz/drift_velocity + np.repeat(time, electrons)
        nphotons  = np.round(np.random.normal(el_gain, el_gain_sigma, size=electrons.sum())).astype(np.int32)
        return dx, dy, dz, dtimes, nphotons
    return simulate_ielectrons

def s1_times_simulator(s1_lighttable, ws):
    S1_LT = create_lighttable_function(os.path.expandvars(s1_lighttable))
    def simulate_S1_times(x, y, z, time, energy):
        s1_photons = np.random.poisson(energy / ws)
        s1_pes_at_pmts = compute_S1_pes_at_pmts(x, y, z, s1_photons, S1_LT)
        s1times = generate_S1_times_from_pes(s1_pes_at_pmts, time)
        return s1times
    return simulate_S1_times

def buffer_times_and_length_getter(wf_pmt_bin_width, wf_sipm_bin_width, el_gap, el_dv, S2tmax=np.inf):
    def get_buffer_times_and_length(S1times, S2times):
        all_times = np.concatenate(S1times).ravel()
        tmin = min(all_times) if len(all_times) else 0
        tmax = max(S2times) + max(wf_sipm_bin_width, wf_pmt_bin_width) + el_gap/el_dv
        buffer_length = np.ceil((tmax-tmin)/wf_sipm_bin_width)*wf_sipm_bin_width
        return tmin, tmax, buffer_length
    return get_buffer_times_and_length

def s1_waveforms_creator(wf_pmt_bin_width):
    def create_s1_waveforms(S1times, tmin, buffer_length):
        s1_wfs = create_S1_waveforms(S1times, buffer_length, wf_pmt_bin_width, tmin)
        return s1_wfs
    return create_s1_waveforms


def s2_waveform_creator (sns_bin_width, LT, EL_drift_velocity):
    def create_s2_waveform(xs, ys, ts, phs, tmin, buffer_length):
        ts_aux = ts-tmin #shift absolute time to start at tmin
        waveforms = electron_loop(xs, ys, ts_aux, phs, LT, EL_drift_velocity, sns_bin_width, buffer_length)
        return np.random.poisson(waveforms)
    return create_s2_waveform

def bin_edges_getter(wf_pmt_bin_width, wf_sipm_bin_width):
    def get_bin_edges(pmt_wfs, sipm_wfs):
        pmt_bins  = np.arange(0, pmt_wfs .shape[1])* wf_pmt_bin_width
        sipm_bins = np.arange(0, sipm_wfs.shape[1])*wf_sipm_bin_width
        return pmt_bins, sipm_bins
    return get_bin_edges
    
    
    

