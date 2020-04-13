"""
-----------------------------------------------------------------------
                                 Hypathia
-----------------------------------------------------------------------

From ancient Greek ‘Υπατια: highest, supreme.

This city reads true waveforms from detsim and compute pmaps from them
without simulating the electronics. This includes:
    - Rebin 1-ns waveforms to 25-ns waveforms to match those produced
      by the detector.
    - Produce a PMT-summed waveform.
    - Apply a threshold to the PMT-summed waveform.
    - Find pulses in the PMT-summed waveform.
    - Match the time window of the PMT pulse with those in the SiPMs.
    - Build the PMap object.
"""
import numpy  as np
import tables as tb

from functools import partial

from .. database       import load_db
from .. core.system_of_units_c import units

from .. reco                  import sensor_functions     as sf
from .. reco                  import tbl_functions        as tbl
from .. reco                  import peak_functions       as pkf
from .. core. random_sampling import NoiseSampler         as SiPMsNoiseSampler
from .. io  .        pmaps_io import          pmap_writer
from .. io  .       mcinfo_io import       mc_info_writer
from .. io  .run_and_event_io import run_and_event_writer
from .. io  .      trigger_io import       trigger_writer
from .. io  . event_filter_io import  event_filter_writer

from .. dataflow            import dataflow as fl
from .. dataflow.dataflow   import push
from .. dataflow.dataflow   import pipe
from .. dataflow.dataflow   import sink

from .  components import city
from .  components import print_every
from .  components import collect
from .  components import copy_mc_info
from .  components import zero_suppress_wfs
from .  components import WfType
from .  components import sensor_data
from .  components import wf_from_files
from .  irene import check_empty_pmap
from .  irene import check_nonempty_indices
from .  irene import get_number_of_active_pmts
from .  irene import build_pmap


@city
def hypathia(files_in, file_out, compression, event_range, print_mod, detector_db, run_number,
             sipm_noise_cut, filter_padding, thr_sipm, thr_sipm_type, pmt_wfs_rebin, pmt_pe_rms,
             s1_lmin, s1_lmax, s1_tmin, s1_tmax, s1_rebin_stride, s1_stride, thr_csum_s1,
             s2_lmin, s2_lmax, s2_tmin, s2_tmax, s2_rebin_stride, s2_stride, thr_csum_s2, thr_sipm_s2,
             pmt_samp_wid=25*units.ns, sipm_samp_wid=1*units.mus):
    if   thr_sipm_type.lower() == "common":
        # In this case, the threshold is a value in pes
        sipm_thr = thr_sipm

    elif thr_sipm_type.lower() == "individual":
        # In this case, the threshold is a percentual value
        noise_sampler = SiPMsNoiseSampler(detector_db, run_number)
        sipm_thr      = noise_sampler.compute_thresholds(thr_sipm)

    else:
        raise ValueError(f"Unrecognized thr type: {thr_sipm_type}. "
                          "Only valid options are 'common' and 'individual'")

    #### Define data transformations
    sd = sensor_data(files_in[0], WfType.mcrd)

    # Raw WaveForm to Corrected WaveForm
    mcrd_to_rwf      = fl.map(rebin_pmts(pmt_wfs_rebin),
                              args = "pmt",
                              out  = "ccwfs")

    # Compute pmt sum
    pmt_sum          = fl.map(pmts_sum, args = 'ccwfs',
                              out  = 'pmt_sum')

    # Find where waveform is above threshold
    zero_suppress    = fl.map(zero_suppress_wfs(thr_csum_s1, thr_csum_s2),
                              args = ("pmt_sum", "pmt_sum"),
                              out  = ("s1_indices", "s2_indices", "s2_energies"))

    # Build the PMap
    compute_pmap     = fl.map(build_pmap(detector_db, run_number, pmt_samp_wid, sipm_samp_wid,
                                         s1_lmax, s1_lmin, s1_rebin_stride, s1_stride, s1_tmax, s1_tmin,
                                         s2_lmax, s2_lmin, s2_rebin_stride, s2_stride, s2_tmax, s2_tmin, thr_sipm_s2),
                              args = ("ccwfs", "s1_indices", "s2_indices", "sipm"),
                              out  = "pmap")


    ### Define data filters

    # Filter events without signal over threshold
    indices_pass    = fl.map(check_nonempty_indices, args = ("s1_indices", "s2_indices"), out = "indices_pass")
    empty_indices   = fl.count_filter(bool, args = "indices_pass")

    # Filter events with zero peaks
    pmaps_pass      = fl.map(check_empty_pmap, args = "pmap", out = "pmaps_pass")
    empty_pmaps     = fl.count_filter(bool, args = "pmaps_pass")

    event_count_in  = fl.spy_count()
    event_count_out = fl.spy_count()

    evtnum_collect  = collect()

    with tb.open_file(file_out, "w", filters = tbl.filters(compression)) as h5out:

        write_event_info_   = run_and_event_writer(h5out)
        write_pmap_         = pmap_writer         (h5out, compression=compression)
        write_trigger_info_ = trigger_writer      (h5out, get_number_of_active_pmts(detector_db, run_number))
        write_indx_filter_  = event_filter_writer (h5out, "s12_indices", compression=compression)
        write_pmap_filter_  = event_filter_writer (h5out, "empty_pmap" , compression=compression)

        # ... and make them sinks
        write_event_info   = sink(write_event_info_  , args=(   "run_number",     "event_number", "timestamp"   ))
        write_pmap         = sink(write_pmap_        , args=(         "pmap",     "event_number"                ))
        write_trigger_info = sink(write_trigger_info_, args=( "trigger_type", "trigger_channels"                ))
        write_indx_filter  = sink(write_indx_filter_ , args=(                     "event_number", "indices_pass"))
        write_pmap_filter  = sink(write_pmap_filter_ , args=(                     "event_number",   "pmaps_pass"))

        result = push(source = wf_from_files(files_in, WfType.mcrd),
                      pipe   = pipe(fl.slice(*event_range, close_all=True),
                                    print_every(print_mod),
                                    event_count_in.spy,
                                    mcrd_to_rwf,
                                    pmt_sum,
                                    zero_suppress,
                                    indices_pass,
                                    fl.branch(write_indx_filter),
                                    empty_indices.filter,
                                    compute_pmap,
                                    pmaps_pass,
                                    fl.branch(write_pmap_filter),
                                    empty_pmaps.filter,
                                    event_count_out.spy,
                                    fl.branch("event_number", evtnum_collect.sink),
                                    fl.fork(write_pmap,
                                            write_event_info,
                                            write_trigger_info)),
                      result = dict(events_in   = event_count_in .future,
                                    events_out  = event_count_out.future,
                                    evtnum_list = evtnum_collect .future,
                                    over_thr    = empty_indices  .future,
                                    full_pmap   = empty_pmaps    .future))

        if run_number <= 0:
            copy_mc_info(files_in, h5out, result.evtnum_list)

        return result


def rebin_pmts(rebin_stride):
    def rebin_pmts(rwf):
        rebinned_wfs = rwf
        if rebin_stride > 1:
            # dummy data for times and widths
            times     = np.zeros(rwf.shape[1])
            widths    = times
            waveforms = rwf
            _, _, rebinned_wfs = pkf.rebin_times_and_waveforms(times, widths, waveforms, rebin_stride=rebin_stride)
        return rebinned_wfs
    return rebin_pmts


def pmts_sum(rwfs):
    return rwfs.sum(axis=0)

