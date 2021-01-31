"""
-----------------------------------------------------------------------
                              Beersheba
-----------------------------------------------------------------------
Beersheba, a city suspended from the heavens, inhabited only by idealists.
This city interpolates corrected hits and applies Lucy-Richardson deconvolution
to the interpolated signal.
The input is esmeralda output containing hits, kdst global information and mc info.
The city outputs :
    - DECO deconvolved hits table
    - MC info (if run number <=0)
    - SUMMARY summary of per event information
"""

import numpy  as np
import tables as tb
import pandas as pd

from os   .path  import expandvars
from scipy.stats import multivariate_normal
from numpy       import nan_to_num

from typing      import Tuple
from typing      import List
from typing      import Callable
from enum        import auto

from .  components import city
from .  components import collect
from .  components import copy_mc_info
from .  components import print_every
from .  components import hits_df_from_files

from .  esmeralda  import summary_writer

from .. reco                   import tbl_functions           as tbl
from .. dataflow               import dataflow                as fl

from .. dataflow.dataflow      import push
from .. dataflow.dataflow      import pipe

from .. database.load_db       import DataSiPM

from .. reco.deconv_functions  import find_nearest
from .. reco.deconv_functions  import cut_and_redistribute_df
from .. reco.deconv_functions  import drop_isolated_sensors
from .. reco.deconv_functions  import deconvolve
from .. reco.deconv_functions  import richardson_lucy
from .. reco.deconv_functions  import InterpolationMethod
from .. reco                import corrections          as cof
from .. io.run_and_event_io    import run_and_event_writer
from .. io.          dst_io    import df_writer
from .. io.          dst_io    import load_dst

from .. evm.event_model        import HitEnergy

from .. types.ic_types         import AutoNameEnumBase

from .. core                   import system_of_units as units


class CutType          (AutoNameEnumBase):
    abs = auto()
    rel = auto()

class DeconvolutionMode(AutoNameEnumBase):
    joint    = auto()
    separate = auto()


def deconvolve_signal(det_db          : pd.DataFrame,
                      psf_fname       : str,
                      e_cut           : float,
                      n_iterations    : int,
                      iteration_tol   : float,
                      sample_width    : List[float],
                      bin_size        : List[float],
                      diffusion       : Tuple[float]=(1., 1., 0.3),
                      deconv_mode     : DeconvolutionMode=DeconvolutionMode.joint,
                      n_dim           : int=2,
                      cut_type        : CutType=CutType.abs,
                      inter_method    : InterpolationMethod=InterpolationMethod.cubic,
                      n_iterations_g  : int=0):

    """
    Applies Lucy Richardson deconvolution to SiPM response with a
    given set of PSFs and parameters.

    Parameters
    ----------
    det_db          : Detector database.
    psf_fname       : Point-spread function.
    e_cut           : Cut in relative value to the max voxel over the deconvolution output.
    n_iterations    : Number of Lucy-Richardson iterations
    iteration_tol   : Stopping threshold (difference between iterations).
    sample_width    : Sampling size of the sensors.
    bin_size        : Size of the interpolated bins.
    deconv_mode     : 'joint' or 'separate', 1 or 2 step deconvolution, see description later.
    diffusion       : Diffusion coefficients in each dimension for 'separate' mode.
    n_dim           : Number of dimensions to apply the method (usually 2).
    cut_type        : Cut mode to the deconvolution output ('abs' or 'rel') using e_cut
                      'abs': cut on the absolute value of the hits.
                      'rel': cut on the relative value (to the max) of the hits.
    inter_method    : Interpolation method.
    n_iterations_g  : Number of Lucy-Richardson iterations for gaussian in 'separate mode'

    Returns
    ----------
    apply_deconvolution : Function that takes hits and returns the
    deconvolved data.
    """
    dimensions    = np.array  (['X', 'Y', 'Z'][:n_dim])
    sample_width  = np.asarray(sample_width           )
    bin_size      = np.asarray(bin_size               )
    diffusion     = np.asarray(diffusion              )

    psfs          = load_dst(psf_fname, 'PSF', 'PSFs')
    det_grid      = [np.arange(det_db[var].min() + bs/2, det_db[var].max() - bs/2 + np.finfo(np.float32).eps, bs)
                     for var, bs in zip(dimensions, bin_size)]
    deconvolution = deconvolve(n_iterations, iteration_tol,
                               sample_width, det_grid       , inter_method)

    if not isinstance(inter_method, InterpolationMethod):
        raise ValueError(f'inter_method {inter_method} is not a valid interpolation method.')
    if not isinstance(cut_type    , CutType            ):
        raise ValueError(f'cut_type {cut_type} is not a valid cut type.')
    if not isinstance(deconv_mode , DeconvolutionMode  ):
        raise ValueError(f'deconv_mode {deconv_mode} is not a valid deconvolution mode.')

    def deconvolve_hits(df, z):
        '''
        Given an slice, applies deconvolution using the PSF
        associated to the passed z.

        Parameters
        ----------
        df : Original input dataframe for the deconvolution (single slice cdst)
        z  : Longitudinal position of the slice.
        Returns
        ----------
        Dataframe with the deconvolved slice.
        '''
        xx, yy = df.Xpeak.unique(), df.Ypeak.unique()
        zz     = z if deconv_mode is DeconvolutionMode.joint else 0
        psf = psfs.loc[(psfs.z == find_nearest(psfs.z, zz)) &
                       (psfs.x == find_nearest(psfs.x, xx)) &
                       (psfs.y == find_nearest(psfs.y, yy)) , :]
        deconv_image, pos = deconvolution(tuple(df.loc[:, dimensions].values.T), df.NormQ.values, psf)

        if   deconv_mode is DeconvolutionMode.joint:
            pass
        elif deconv_mode is DeconvolutionMode.separate:
            dist         = multivariate_normal(np.zeros(n_dim), diffusion**2 * z * units.mm / units.cm) #Z is in mm in cdst
            cols         = tuple(f"{v.lower()}r" for v in dimensions)
            psf_cols     = psf.loc[:, cols]
            gaus         = dist.pdf(psf_cols.values)
            psf          = gaus.reshape(psf_cols.nunique())
            deconv_image = nan_to_num(richardson_lucy(deconv_image, psf, n_iterations_g, iteration_tol))

        return create_deconvolution_df(df, deconv_image.flatten(), pos, cut_type, e_cut, n_dim)

    def apply_deconvolution(df):
        '''
        Given an event cdst, it iterates through its S2s and applies deconvolution
        to each S2.

        Parameters
        ----------
        df : Original input dataframe for the deconvolution (event cdst)

        Returns
        ----------
        Dataframe with the deconvolved event.
        '''
        deco_dst = []
        df.loc[:, "NormQ"] = np.nan
        for peak, hits in df.groupby("npeak"):
            all_hits = []
            for z, df_z in hits.groupby('Z'):
                if len(df_z)==1:
                    all_hits.append(df_z)
                else:
                    df_z.loc[:, "NormQ"] = df_z.Q/df_z.Q.sum()
                    deconv_hits_ = deconvolve_hits(df_z, z)
                    deconv_hits = distribute_energy_z(deconv_hits_, df_z)
                    all_hits.append(deconv_hits)
            all_hits_df = pd.concat(all_hits, ignore_index=True).sort_values(['Z', 'E'])
            deco_dst.append(all_hits_df)

        return pd.concat(deco_dst, ignore_index=True)

    return apply_deconvolution


def create_deconvolution_df(hits, deconv_e, pos, cut_type, e_cut, n_dim):
    '''
    Given the output of the deconvolution, it cuts the low energy voxels and
    creates a dataframe object with the resulting output.

    Parameters
    ----------
    hits     : Original input dataframe for the deconvolution (S2 cdst)
    deconv_e : Deconvolution energy distribution (n-dim array)
    pos      : Position of the deconvolved hits.
    cut_type : CutType object with the cut mode.
    e_cut    : Value for the energy cut.
    n_dim    : Number of dimensions of the deconvolution (tipically 2 as of now)

    Returns
    ----------
    df       : Dataframe with the deconvolution input after energy cutting.
    '''

    df  = pd.DataFrame(columns=['event', 'npeak', 'X', 'Y', 'Z', 'E'])

    if   cut_type is CutType.abs:
        sel_deconv = deconv_e > e_cut
    elif cut_type is CutType.rel:
        sel_deconv = deconv_e / deconv_e.max() > e_cut
    else:
        raise ValueError(f'cut_type {cut_type} is not a valid cut type.')

    if sum(sel_deconv)==0:
        sel_deconv = deconv_e==max(deconv_e)

    df['E']     = deconv_e[sel_deconv]
    df['event'] = hits.event.unique()[0]
    df['npeak'] = hits.npeak.unique()[0]
    df['Z']     = hits.Z    .unique()[0]
    df['X']     = pos[0][sel_deconv]
    df['Y']     = pos[1][sel_deconv]
    df['time'] = hits.time.unique()[0]
    return df

def distribute_energy_z(df, cdst):
    '''
    Assign the energy of a dataframe (cdst) to another dataframe (deconvolved),
    distributing it according to the charge fraction of each deconvolution hit.

    Parameters
    ----------
    df          : Deconvolved dataframe with a single S2 (npeak)
    cdst        : Dataframe with the sensor response (usually a cdst)
    '''

    sum_e = cdst.E.sum()
    sum_edc = df.E.sum()
    df.loc[:,'E'] = df.E/sum_edc*sum_e
    return df


def cut_over_Q(q_cut, redist_var):
    '''
    Apply a cut over the SiPM charge condition to hits and redistribute the
    energy variables.

    Parameters
    ----------
    q_cut      : Charge value over which to cut.
    redist_var : List with variables to be redistributed.

    Returns
    ----------
    cut_over_Q : Function that will cut the dataframe and redistribute
    values.
    '''
    cut = cut_and_redistribute_df(f"Q > {q_cut}", redist_var)

    def cut_over_Q(df):  # df shall be an event cdst
        cdst = df.groupby(['event', 'npeak']).apply(cut).reset_index(drop=True)

        return cdst

    return cut_over_Q


def drop_isolated(distance, redist_var):
    """
    Drops rogue/isolated hits (SiPMs) from hits.

    Parameters
    ----------
    distance   : Sensor pitch.
    redist_var : List with variables to be redistributed.

    Returns
    ----------
    drop_isolated_sensors : Function that will drop the isolated sensors.
    """
    drop = drop_isolated_sensors(distance, redist_var)

    def drop_isolated(df): # df shall be an event cdst
        df = df.groupby(['event', 'npeak']).apply(drop).reset_index(drop=True)

        return df

    return drop_isolated



def hits_redistributor(thr):
    """ Distribute energy of hits with Q below thr"""
    def copy_last_position_NN(dfp):
        """ For NN hits copy X, Y position of the closest non-NN hit"""
        indxs = dfp.groupby('Z').apply(lambda x:x.Q.idxmax()).values
        NN_slices = dfp[dfp.Q<0].index
        non_NN = np.setdiff1d(indxs, NN_slices)
        Z_aux = dfp.loc[non_NN].Z
        NN_z = dfp.loc[NN_slices].Z.values
        closest = np.argmin(np.abs(Z_aux.values[None, :]-NN_z[:, None]), axis=-1)
        dfp.loc[NN_slices, 'X']=dfp.loc[Z_aux.index[closest],'X'].values
        dfp.loc[NN_slices, 'Y']=dfp.loc[Z_aux.index[closest],'Y'].values
        return dfp

    def red_per_z(dfp_z):
        dfp_cp = dfp_z.copy()
        Esum = dfp_cp.E.sum()
        pass_indx = dfp_cp[dfp_cp.Q>thr].index
        if len(pass_indx)>0:
            Qsum = dfp_cp.loc[pass_indx, 'Q'].sum()
            dfp_cp.loc[pass_indx, 'E'] = dfp_cp.loc[pass_indx, 'Q']*Esum/Qsum
            return dfp_cp.loc[pass_indx]
        else:
            maxid = dfp_cp.Q.idxmax
            dfp_cp = dfp_cp.loc[maxid].to_frame().T
            dfp_cp.Q = -dfp_cp.Q
            return dfp_cp
    def redistribute_hits(dfp):
        dfp = copy_last_position_NN(dfp)
        return dfp.groupby('Z').apply(red_per_z).reset_index(drop=True)
    return redistribute_hits


def hits_corrector(map_fname        : str  ,
                   apply_temp       : bool) -> Callable:
    """
    For a given correction map and hit threshold/ merging parameters returns a function that applies thresholding, merging and
    energy and Z corrections to a given HitCollection object.

    Parameters
    ----------
    map_fname        : string (filepath)
        filename of the map
    apply_temp       : bool
        whether to apply temporal corrections
        must be set to False if no temporal correction dataframe exists in map file

    Returns
    ----------
    A function that takes HitCollection as input and returns HitCollection that containes
    only non NN hits of charge above threshold_charge with modified Ec and Z fields.
    """
    map_fname = expandvars(map_fname)
    maps      = cof.read_maps(map_fname)
    get_coef  = cof.apply_all_correction(maps, apply_temp = apply_temp, norm_strat = cof.norm_strategy.kr)
    if maps.t_evol is not None:
        time_to_Z = cof.get_df_to_z_converter(maps)
    else:
        time_to_Z = lambda x: x
    def threshold_and_correct_hits(hitc : pd.DataFrame) -> pd.DataFrame:
        t = hitc.time
        Ec = hitc.E * get_coef(hitc.X,hitc.Y,hitc.Z,hitc.time)
        hitc.loc[:, 'Ec'] = Ec
        hitc.loc[:, 'Zc'] = time_to_Z(hitc.Z)
        return hitc
    return threshold_and_correct_hits




def check_nonempty_dataframe(df) -> bool:
    """
    Filter for Beersheba flow. The flow stops if:
        1. there are no hits (after droping isolated sensors)
    """
    return len(df) > 0


def deconv_writer(h5out, compression='ZLIB4'):
    """
    For a given open table returns a writer for deconvolution hits dataframe
    """
    def write_deconv(df):
        return df_writer(h5out              = h5out             ,
                         df                 = df                ,
                         compression        = compression       ,
                         group_name         = 'DECO'            ,
                         table_name         = 'Events'          ,
                         descriptive_string = 'Deconvolved hits',
                         columns_to_index   = ['event']         )
    return write_deconv
def kdst_from_df_writer(h5out, compression='ZLIB4'):
    """
    For a given open table returns a writer for KDST dataframe info
    """
    def write_kdst(df):
        return df_writer(h5out              = h5out        ,
                         df                 = df           ,
                         compression        = compression  ,
                         group_name         = 'DST'        ,
                         table_name         = 'Events'     ,
                         descriptive_string = 'KDST Events',
                         columns_to_index   = ['event']    )
    return write_kdst



@city
def beersheba(files_in, file_out, compression, event_range, print_mod, detector_db, run_number,
              deconv_params = dict(), cor_params=dict()):
    """
    The city corrects Penthesilea hits energy and extracts topology information.
    ----------
    Parameters
    ----------
    files_in    : str, filepath
         Input file
    file_out    : str, filepath
         Output file
    compression : str
         Default  'ZLIB4'
    event_range : int /'all_events'
         Number of events from files_in to process
    print_mode  : int
         How frequently to print events
    run_number  : int
         Has to be negative for MC runs

    deconv_params : dict
        q_cut          : float
            Minimum charge (pes) on a hit (SiPM)
        drop_dist      : float
            Distance to check if a SiPM is isolated
        psf_fname      : string (filepath)
            Filename of the psf
        e_cut          : float
            Cut over the deconvolution output, arbitrary units (order 1e-3)
        n_iterations   : int
            Number of iterations to be applied if the iteration_tol criteria
            is not fulfilled before.
        iteration_tol  : float
            Stopping threshold (difference between iterations). I
        sample_width   : list[float]
            Sampling of the sensors in each dimension (usuallly the pitch).
        bin_size       : list[float]
            Bin size (mm) of the deconvolved image.
        deconv_mode    : str ('joint', 'separate')
            - 'joint' deconvolves once using a PSF based on Z that includes
               both EL and diffusion spread aproximated to a Z range.
            - 'separate' deconvolves twice, first using the EL PSF, then using
               a gaussian PSF based on the exact Z position of the slice.
        diffusion      : tuple(float)
            Diffusion coefficients in each dimmension (mm/sqrt(cm))
            used if deconv_mode is 'separate'
        n_dim          : int
            Number of dimensions used in deconvolution, currently only 2 max:
            n_dim = 2 -> slice by slice XY deconvolution.
            n_dim = 3 -> XYZ deconvolution (in the works).
        inter_method   : str (None, 'linear', 'cubic')
            Sensor interpolation method. If None, no interpolation will be applied.
            'cubic' not supported for 3D deconvolution.
        n_iterations_g : int
            Number of Lucy-Richardson iterations for gaussian in 'separate mode'

    ----------
    Input
    ----------
    Esmeralda output
    ----------
    Output
    ----------
    DECO    : Deconvolved hits table
    MC info : (if run number <=0)
    SUMMARY : Table with the summary from Esmeralda.
"""

    deconv_params['cut_type'    ] = CutType            (deconv_params['cut_type'    ])
    deconv_params['deconv_mode' ] = DeconvolutionMode  (deconv_params['deconv_mode' ])
    deconv_params['inter_method'] = InterpolationMethod(deconv_params['inter_method'])

    deconv_params['psf_fname'   ] = expandvars(deconv_params['psf_fname'])

    for p in ['sample_width', 'bin_size', 'diffusion']:
        if len(deconv_params[p]) != deconv_params['n_dim']:
            raise ValueError         (f"Parameter {p} dimensions do not match n_dim parameter")
    if deconv_params['n_dim'] > 2:
        raise     NotImplementedError(f"{deconv_params['n_dim']}-dimensional PSF not yet implemented")

    # cut_sensors           = fl.map(cut_over_Q   (deconv_params.pop("q_cut")    , ['E', 'Ec']),
    #                                item = 'cdst')
    # drop_sensors          = fl.map(drop_isolated(deconv_params.pop("drop_dist"), ['E', 'Ec']),
    #                                item = 'cdst')

    cut_sensors           = fl.map(hits_redistributor  (deconv_params.pop("q_cut")),
                                   item = 'hits')


    filter_events_no_hits = fl.map(check_nonempty_dataframe,
                                   args = 'hits',
                                   out  = 'hits_passed_no_hits')
    deconvolve_events     = fl.map(deconvolve_signal(DataSiPM(detector_db, run_number), **deconv_params),
                                   args = 'hits',
                                   out  = 'deconv_dst')
    correct_hits = fl.map(hits_corrector( **cor_params),
                          item = 'deconv_dst')

    event_count_in        = fl.spy_count()
    event_count_out       = fl.spy_count()
    events_passed_no_hits = fl.count_filter(bool, args = "hits_passed_no_hits")

    evtnum_collect        = collect()

    with tb.open_file(file_out, "w", filters = tbl.filters(compression)) as h5out:
        # Define writers
        write_event_info = fl.sink(run_and_event_writer (h5out), args = ("run_number", "event_number", "timestamp"))
        write_deconv     = fl.sink(  deconv_writer(h5out=h5out), args =  "deconv_dst")
        write_kdst_table      = fl.sink( kdst_from_df_writer(h5out), args="kdst")
        result = push(source = hits_df_from_files(files_in),
                      pipe   = pipe(fl.slice(*event_range, close_all=True)    ,
                                    print_every(print_mod)                    ,
                                    event_count_in.spy                        ,
                                    cut_sensors                               ,
                                    filter_events_no_hits                     ,
                                    events_passed_no_hits    .filter          ,
                                    deconvolve_events                         ,
                                    correct_hits                              ,
                                    event_count_out.spy                       ,
                                    fl.branch("event_number"     ,
                                              evtnum_collect.sink)            ,
                                    fl.fork(write_deconv    ,
                                            write_event_info,
                                            write_kdst_table))                ,
                      result = dict(events_in   = event_count_in       .future,
                                    events_out  = event_count_out      .future,
                                    evtnum_list = evtnum_collect       .future,
                                    events_pass = events_passed_no_hits.future))

        if run_number <= 0:
            copy_mc_info(files_in, h5out, result.evtnum_list,
                         detector_db, run_number)

        return result
