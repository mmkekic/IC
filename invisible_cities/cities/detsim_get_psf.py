import numpy  as np
import tables as tb
import pandas as pd

from typing import Callable

from invisible_cities.core.core_functions  import in_range

from invisible_cities.core import system_of_units as units


def create_lighttable_function(filename):

    lt     = pd.read_hdf(filename, "LightTable")
    Config = pd.read_hdf(filename, "Config")

    signaltype = Config.loc["signal_type"].value
    sensor     = Config.loc["sensor"]     .value

    # drop total column
    lt = lt.drop(columns = [sensor + "_total"])

    # check sensors order
    sensors = lt.columns.to_list()
    sorted_sensors = sorted(sensors, key=lambda name: int(name.split("_")[-1]))
    lt = lt.loc[:, sorted_sensors]

    nsensors = len(sorted_sensors)

    if signaltype == "S2":

        xcenters = np.sort(np.unique(lt.index.get_level_values('x')))
        ycenters = np.sort(np.unique(lt.index.get_level_values('y')))

        xbins=binedges_from_bincenters(xcenters)
        ybins=binedges_from_bincenters(ycenters)

        def get_lt_values(xs, ys):
            xindices = pd.cut(xs, xbins, labels=xcenters)
            yindices = pd.cut(ys, ybins, labels=ycenters)
            indices = pd.Index(zip(xindices, yindices), name=("x", "y"))

            mask = indices.isin(lt.index)

            values = np.zeros((len(xs), nsensors))
            values[mask] = lt.loc[indices[mask]]
            return values

    elif signaltype == "S1":

        xcenters = np.sort(np.unique(lt.index.get_level_values('x')))
        ycenters = np.sort(np.unique(lt.index.get_level_values('y')))
        zcenters = np.sort(np.unique(lt.index.get_level_values('z')))

        xbins=binedges_from_bincenters(xcenters)
        ybins=binedges_from_bincenters(ycenters)
        zbins=binedges_from_bincenters(zcenters)

        def get_lt_values(xs, ys, zs):

            xindices = pd.cut(xs, xbins, labels=xcenters)
            yindices = pd.cut(ys, ybins, labels=ycenters)
            zindices = pd.cut(zs, zbins, labels=zcenters)
            indices = pd.Index(zip(xindices, yindices, zindices), name=("x", "y", "z"))

            mask = indices.isin(lt.index)
            values = np.zeros((len(xs), nsensors))
            values[mask] = lt.loc[indices[mask]]
            return values

    return get_lt_values


def binedges_from_bincenters(bincenters: np.ndarray)->np.ndarray:
    """
    computes bin-edges from bin-centers. The extremes of the edges are asigned to
    the extremes of the bin centers.

    Parameters:
        :bincenters: np.ndarray
            bin centers
    Returns:
        :binedges: np.ndarray
            bin edges
    """
    binedges = np.zeros(len(bincenters)+1)

    binedges[1:-1] = (bincenters[1:] + bincenters[:-1])/2.
    binedges[0]  = bincenters[0]
    binedges[-1] = bincenters[-1]

    return binedges
