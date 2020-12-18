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
from .. detsim.light_tables_c     import LT_SiPM
from .. detsim.light_tables_c     import LT_PMT
from .. detsim.s2_waveforms_c     import create_wfs
from .. detsim.detsim_waveforms   import s1_waveforms_creator

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
    lt_pmt  = LT_PMT (fname=os.path.expandvars(s2_lighttable))
    lt_sipm = LT_SiPM(fname=os.path.expandvars(sipm_psf), sipm_database=datasipm)
    el_gap = lt_sipm.el_gap
    el_gain_sigma = np.sqrt(el_gain * conde_policarpo_factor)


    select_s1_candidate_hits = fl.map(hits_selector(False),
                                item = ('x', 'y', 'z', 'energy', 'time', 'label'))

    select_active_hits = fl.map(hits_selector(True),
                                args = ('x', 'y', 'z', 'energy', 'time', 'label'),
                                out = ('x_a', 'y_a', 'z_a', 'energy_a', 'time_a', 'labels_a'))



    filter_events_no_active_hits = fl.map (lambda x:np.any(x),
                                           args= 'energy_a',
                                           out = 'passed')
    events_passed_active_hits = fl.count_filter(bool, args='passed')

    simulate_electrons = fl.map(ielectron_simulator(wi, fano_factor, lifetime, transverse_diffusion, longitudinal_diffusion, drift_velocity, el_gain, el_gain_sigma),
                                args = ('x_a', 'y_a', 'z_a', 'time_a', 'energy_a'),
                                out  = ('x_ph', 'y_ph', 'z_ph', 'times_ph', 'nphotons'))

    get_buffer_times_and_length = fl.map(buffer_times_and_length_getter(wf_pmt_bin_width, wf_sipm_bin_width, el_gap, drift_velocity_EL, max_length=max_time),
                                         args = ('time', 'times_ph'),
                                         out = ('tmin', 'buffer_length'))

    create_pmt_s1_waveforms = fl.map(s1_waveforms_creator(s1_lighttable, ws, wf_pmt_bin_width),
                                     args = ('x_a', 'y_a', 'z_a', 'time_a', 'energy_a', 'tmin', 'buffer_length'),
                                     out = 's1_pmt_waveforms')
    create_pmt_s2_waveforms = fl.map(s2_waveform_creator (wf_pmt_bin_width, lt_pmt, drift_velocity_EL),
                                     args = ('x_ph', 'y_ph', 'times_ph', 'nphotons', 'tmin', 'buffer_length'),
                                     out = 's2_pmt_waveforms')
    sum_pmt_waveforms = fl.map(lambda x, y : x+y,
                                  args = ('s1_pmt_waveforms', 's2_pmt_waveforms'),
                                  out = 'pmt_bin_wfs')
    create_pmt_waveforms = fl.pipe(create_pmt_s1_waveforms, create_pmt_s2_waveforms, sum_pmt_waveforms)

    create_sipm_waveforms = fl.map(s2_waveform_creator (wf_sipm_bin_width, lt_sipm, drift_velocity_EL),
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
                                         get_buffer_times_and_length,
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
def hits_selector (active_only=True):
    def select_hits(x, y, z, energy, time, label):
        sel = (label == "ACTIVE")
        if not active_only:
            sel =  sel or (label == "BUFFER")
        return x[sel], y[sel], z[sel], energy[sel], time[sel], label[sel]
    return select_hits


def ielectron_simulator(wi, fano_factor, lifetime, transverse_diffusion, longitudinal_diffusion, drift_velocity, el_gain, el_gain_sigma):
    def simulate_ielectrons(x, y, z, time, energy):        
        electrons = generate_ionization_electrons(energy, wi, fano_factor)
        electrons = drift_electrons(z, electrons, lifetime, drift_velocity)
        dx, dy, dz = diffuse_electrons(x, y, z, electrons, transverse_diffusion, longitudinal_diffusion)
        dtimes  = dz/drift_velocity + np.repeat(time, electrons)
        nphotons  = np.round(np.random.normal(el_gain, el_gain_sigma, size=electrons.sum())).astype(np.int32)
        return dx, dy, dz, dtimes, nphotons
    return simulate_ielectrons

def buffer_times_and_length_getter(wf_pmt_bin_width, wf_sipm_bin_width, el_gap, el_dv, max_length):
    max_sensor_bin = max(wf_pmt_bin_width, wf_sipm_bin_width)
    def get_buffer_times_and_length(times_a, times_ph):
        start_time = np.floor(min(times_a)/max_sensor_bin)*max_sensor_bin
        el_traverse_time = el_gap/el_dv
        end_time   = np.ceil((max(times_ph)+el_traverse_time)/max_sensor_bin)*max_sensor_bin
        buffer_length = min(max_length, end_time-start_time)
        return start_time, buffer_length
    return get_buffer_times_and_length



def s2_waveform_creator (sns_bin_width, LT, EL_drift_velocity):
    def create_s2_waveform(xs, ys, ts, phs, tmin, buffer_length):
        waveforms = create_wfs(xs, ys, ts, phs, LT, EL_drift_velocity, sns_bin_width, buffer_length, tmin)
        return np.random.poisson(waveforms)
    return create_s2_waveform

def bin_edges_getter(wf_pmt_bin_width, wf_sipm_bin_width):
    def get_bin_edges(pmt_wfs, sipm_wfs):
        pmt_bins  = np.arange(0, pmt_wfs .shape[1])* wf_pmt_bin_width
        sipm_bins = np.arange(0, sipm_wfs.shape[1])*wf_sipm_bin_width
        return pmt_bins, sipm_bins
    return get_bin_edges
    
    
    

