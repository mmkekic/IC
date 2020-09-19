import numpy as np
import scipy
from typing import Callable
import pandas as pd
import line_profiler
from ..core.core_functions import in_range
from ..detsim.detsim_loop  import electron_loop
##################################
######### WAVEFORMS ##############
##################################

def create_waveform(times    : np.ndarray,
                    pes      : np.ndarray,
                    bins     : np.ndarray,
                    nsamples : int) -> np.ndarray:
    """
    This function builds a waveform from a set of (buffer_time, pes) values.
    This set is of values come from the times and pes arguments.

    Parameters:
        :times: np.ndarray
            a vector with the buffer times in which a photoelectron is produced in the detector.
        :pes: np.ndarray
            a vector with the photoelectrons produced in the detector in
            each of the buffer times in times argument.
        :bins: np.ndarray
            a vector with the output waveform bin times (for example [0, 25, 50, ...] if
            the detector has a sampling time of 25).
        :nsamples: int
            an integer that controlls the distribution of the photoelectrons in each of
            the waveform bins. The counts (N) in a given time bin (T) are uniformly distributed
            between T and the subsequent nsamples-1
            nsamples must be >=1 an <len(bins).
    Returns:
        :wf: np.ndarray
            waveform
    """
    if (nsamples<1) or (nsamples>len(bins)):
        raise ValueError("nsamples must lay betwen 1 and len(bins) (inclusive)")

    wf = np.zeros(len(bins)-1 + nsamples-1)
    if np.sum(pes.data)==0:
        return wf[:len(bins)-1]

    t = np.repeat(times, pes)
    sel = in_range(t, bins[0], bins[-1])

    indexes = np.digitize(t[sel], bins)-1
    indexes, counts = np.unique(indexes, return_counts=True)

    i_sample = np.arange(nsamples)
    for index, c in zip(indexes, counts):
        idxs = np.random.choice(i_sample, size=c)
        idx, sp = np.unique(idxs, return_counts=True)
        wf[index + idx] += sp
    return wf[:len(bins)-1]


def create_pmt_waveforms(signal_type   : str,
                         buffer_length : float,
                         bin_width     : float) -> Callable:
    """
    This function calls recursively to create_waveform. See create_waveform for
    an explanation of the arguments not explained below.

    Parameters
        :pes_at_sensors:
            an array with size (#sensors, len(times)). It is the same
            as pes argument in create_waveform but for each sensor in axis 0.
        :wf_buffer_time:
            a float with the waveform extent (in default IC units)
        :bin_width:
            a float with the time distance between bins in the waveform buffer.
    Returns:
        :create_sensor_waveforms_: function
    """
    bins = np.arange(0, buffer_length + bin_width, bin_width)

    if signal_type=="S1":

        def create_pmt_waveforms_(S1times : list):
            wfs = np.stack([np.histogram(times, bins=bins)[0] for times in S1times])
            return wfs

    elif signal_type=="S2":

        def create_pmt_waveforms_(times          : np.ndarray,
                                  pes_at_sensors : np.ndarray,
                                  nsamples       : int = 1):
            wfs = np.stack([create_waveform(times, pes, bins, nsamples) for pes in pes_at_sensors])
            return wfs
    else:
        ValueError("signal_type must be one of S1 or S1")

    return create_pmt_waveforms_

import line_profiler
import atexit
profile = line_profiler.LineProfiler()
atexit.register(profile.print_stats)

def create_sipm_waveforms(wf_buffer_length  : float,
                          wf_sipm_bin_width : float,
                          datasipm : pd.DataFrame,
                          psf,
                          drift_velocity_EL : float):
    nlen = wf_buffer_length//wf_sipm_bin_width
    @profile
    def create_sipm_waveforms_(times,
                               photons,
                               dx,
                               dy):
        sipmwfs = electron_loop(dx, dy, times, photons.astype(np.uint),  psf, drift_velocity_EL, wf_sipm_bin_width, nlen)
        sipmwfs = np.random.poisson(sipmwfs)
        return sipmwfs

    return create_sipm_waveforms_
