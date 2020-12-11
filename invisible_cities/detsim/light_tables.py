import numpy  as np
import pandas as pd

from typing import Callable

from functools import partial

from .. core.core_functions import binedges_from_bincenters

from .. io.dst_io  import load_dst


def create_lighttable_function(filename : str)->Callable:
    """From a lighttable file, it returns a function of (x, y) for S2 signal
    or (x, y, z) for S1 signal type. Signal type is read from the table.
    Parameters:
        :filename: str
            name of the lighttable file
    Returns:
        :get_lt_values: Callable
            this is a function which access the desired value inside
            the lighttable. The lighttable values would be the nearest
            points to the input positions. If the input positions are
            outside the lighttable boundaries, zero is returned.
            Input values must be vectors of same lenght, I. The output
            shape will be (I, number_of_pmts).
    """
    lt     = load_dst(filename, "LT", "LightTable")
    Config = load_dst(filename, "LT", "Config")    .set_index("parameter")
    sensor = Config.loc["sensor"].value
    act_r  = float(Config.loc["ACTIVE_rad"].value)
    lt     = lt.drop(sensor + "_total", axis=1) # drop total column

    def get_lt_values(xs, ys, zs):
        if len(zs) == 1:
            zs = np.full(len(xs), zs)
        if not (len(xs) == len(ys) == len(zs)):
            raise Exception("input arrays must be of same shape")
        sel = (np.sqrt(xs**2 + ys**2) <= act_r) & (zbins[0]<=zs) & (zs<=zbins[-1]) #inside bins
        xindices = pd.cut(xs[sel], xbins, include_lowest=True, labels=xcenters)
        yindices = pd.cut(ys[sel], ybins, include_lowest=True, labels=ycenters)
        zindices = pd.cut(zs[sel], zbins, include_lowest=True, labels=zcenters)
        indices  = pd.Index(zip(xindices, yindices, zindices), name=("x", "y", "z"))
        values   = np.zeros((len(xs), nsensors))
        values[sel] = lt.loc[indices]
        return values

    if lt.get("z") is None:
        lt.loc[:, "z"] = 1 # add fake z
        had_z = False
    else:
        had_z = True

    lt = lt.set_index(["x", "y", "z"])
    nsensors = lt.shape[-1]

    xcenters = np.unique(lt.index.get_level_values('x'))
    ycenters = np.unique(lt.index.get_level_values('y'))
    zcenters = np.unique(lt.index.get_level_values('z'))

    xbins = binedges_from_bincenters(xcenters, range=(-act_r, act_r))
    ybins = binedges_from_bincenters(ycenters, range=(-act_r, act_r))
    zbins = binedges_from_bincenters(zcenters)

    if had_z: return get_lt_values
    else    : return partial(get_lt_values, zs=np.array([1]))
