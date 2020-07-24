import numpy as np

from typing import List
from typing import Tuple

from invisible_cities.cities.detsim_get_psf import binedges_from_bincenters

#######################################
######### ELECTRON SIMULATION #########
#######################################
def generate_ionization_electrons(wi          : float,
                                  fano_factor : float,
                                  energies    : np.ndarray) -> np.ndarray:
    """ generate ionization secondary electrons from energy deposits

    Parrameters:
        :wi: float
            ionization yield
        :fano_factor: float
            fano-factor. related with the deviation in ionization electrons
        :energies: np.ndarray
            energy hits
    Returns:
        :nes: np.ndarray
            the ionization electrons per hit
    """
    nes  = np.array(energies/wi, dtype = int)
    pois = nes < 10
    nes[ pois] = np.random.poisson(nes[pois])
    nes[~pois] = np.round(np.random.normal(nes[~pois], np.sqrt(nes[~pois] * fano_factor)))
    return nes


def drift_electrons(lifetime       : float,
                    drift_velocity : float,
                    zs             : np.ndarray,
                    electrons      : np.ndarray) -> np.array:
    """ returns number of electrons due to lifetime loses from secondary electrons

    Parameters:
        :lifetime: float
            electron lifetime
        :drif_velocity: float
            drif velocity at the active volume of the detector
        :zs:
            the z hits
        :electrons:
    Returns:
        :nes: np.ndarray
            the drifted electrons per initial hit
    """
    ts  = zs / drift_velocity
    nes = electrons - np.random.poisson(electrons * (1. - np.exp(-ts/lifetime)))
    nes[nes < 0] = 0
    return nes


def diffuse_electrons(transverse_diffusion   : float,
                      longitudinal_diffusion : float,
                      xs                     : np.ndarray,
                      ys                     : np.ndarray,
                      zs                     : np.ndarray,
                      electrons              : np.ndarray)\
                      -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    starting from hits with positions xs, ys, zs, and number of electrons,
    apply diffusion and return diffused positions xs, ys, zs for each electron.

    Paramters:
        :transverse_diffusion: float
        :longitudinal_diffusion: float
        :xs, ys, zs: np.ndarray (1D of size: #hits)
            postions of initial hits
        :electrons:
            electrons at the EL before drifting per hit
    Returns:
        :dxs, dys, dzs: np.ndarray (1D of size: #hits x #total electrons)
            diffused positions
    """
    xs = np.repeat(xs, electrons.astype(int))
    ys = np.repeat(ys, electrons.astype(int))
    zs = np.repeat(zs, electrons.astype(int))

    # substitute z<0 to z=0
    sel = zs<0
    zs[sel] = 0

    sqrtz = zs ** 0.5
    dxs  = np.random.normal(xs, sqrtz * transverse_diffusion)
    dys  = np.random.normal(ys, sqrtz * transverse_diffusion)
    dzs  = np.random.normal(zs, sqrtz * longitudinal_diffusion)

    return (dxs, dys, dzs)


def voxelize(voxel_size : list,
             dx         : np.ndarray,
             dy         : np.ndarray,
             dz         : np.ndarray)\
             ->Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    voxelization of the electrons that reach the EL

    Parameters:
        :voxel_size: list, array-like (1D of size: 3)
            voxel sizes in mm
    Returns:
        :dx, dy, dz:
            voxelized positions
        :nes:
            electrons in each position
    """
    if voxel_size:
        ####### HISTOGRAM dx, dy, dz #######
        xmin, xmax = np.min(dx), np.max(dx)
        ymin, ymax = np.min(dy), np.max(dy)
        zmin, zmax = np.min(dz), np.max(dz)

        xcenters = np.arange(xmin, xmax + voxel_size[0], voxel_size[0])
        ycenters = np.arange(ymin, ymax + voxel_size[1], voxel_size[1])
        zcenters = np.arange(zmin, zmax + voxel_size[2], voxel_size[2])

        xbins = binedges_from_bincenters(xcenters)
        ybins = binedges_from_bincenters(ycenters)
        zbins = binedges_from_bincenters(zcenters)

        H, _ = np.histogramdd((dx, dy, dz), bins=[xbins, ybins, zbins])

        ######### RECOVER dx, dy, dz with updated nes ######
        dx, dy, dz = np.meshgrid(xcenters, ycenters, zcenters, indexing="ij")
        dx, dy, dz = dx.flatten(), dy.flatten(), dz.flatten()
        nes = H.flatten()

        sel = nes>0
        dx, dy, dz, nes = dx[sel], dy[sel], dz[sel], nes[sel]

        return dx, dy, dz, nes

    else:
        return dx, dy, dz, np.ones(dx.shape)
