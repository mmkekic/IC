import os
import tables as tb
import numpy  as np
import pandas as pd
import sparse
import copy
import operator
import itertools

from cvxopt import solvers as cvxopt_solvers
from cvxopt import matrix as cvxopt_matrix
from collections import OrderedDict
from typing      import Callable

from .. reco                import tbl_functions        as tbl
from .. reco                import corrections          as cof
from .. dataflow            import dataflow             as fl
from .. dataflow.dataflow   import push
from .. dataflow.dataflow   import pipe
from .. evm.event_model     import MCInfo
from .  components import city
from .  components import print_every
from .  components import collect
from .  components import copy_mc_info
from .  components import get_run_number
from .  components import get_event_info
from .  components import check_lengths



from .. io.run_and_event_io import run_and_event_writer
from .. io. event_filter_io import event_filter_writer
from .. io.          dst_io import df_writer
from .. io.          dst_io import load_dst

from .. database.load_db       import DataSiPM

from typing      import Iterator
from typing      import Mapping
from typing      import List
from typing      import Dict
from typing      import Union


#numba is throwing trash
import logging
numba_logger = logging.getLogger('numba')
numba_logger.setLevel(logging.WARNING)

from IPython.core.debugger import set_trace
        
class CVXOPTQPSolver(object):
    def __init__(self, lambd):
        self.lambda_val = lambd
    def solve(self, A, y, x0):
        n = A.shape[0]
        m = A.shape[1]
        P = cvxopt_matrix(np.dot(A.transpose(), A))
        id_v = np.ones(m)
        q = cvxopt_matrix(self.lambda_val*id_v - np.dot(A.transpose(), y))
        b = cvxopt_matrix(np.zeros(m))
        Q = cvxopt_matrix(-np.eye(m, dtype=float))
        sol = cvxopt_solvers.qp(P = P, q=q, G=Q, h=b, initvals = x0, options={'show_progress': False})
        xnew = np.array(sol['x']).squeeze()
        return xnew

def hitsdf_and_kdst_from_files(paths: List[str]) -> Iterator[Dict[str,Union[pd.DataFrame, pd.DataFrame, MCInfo, int, float]]]:
    for path in paths:
        try:
#            hits_df = load_dst (path, 'RECO', 'Events')
            kdst_df = load_dst (path, 'DST' , 'Events')
        except tb.exceptions.NoSuchNodeError:
            continue

        with tb.open_file(path, "r") as h5in:
            try:
                run_number  = get_run_number(h5in)
                event_info  = get_event_info(h5in)
                hits_table  = h5in.root.RECO.Events
            except (tb.exceptions.NoSuchNodeError, IndexError):
                continue

            #check_lengths(event_info, hits_df.event.unique())

            for evtinfo in event_info:
                event_number, timestamp = evtinfo.fetch_all_fields()
                hits = pd.DataFrame.from_records(hits_table.read_where( 'event == {}'.format(event_number)))
                if len(hits)>0:
                    yield dict(hits = hits,
                               kdst = kdst_df.loc[kdst_df.event==event_number],
                               run_number = run_number,
                               event_number = event_number,
                               timestamp = timestamp)
                else:
                    continue

def sipm_indx_getter(sipm_df):
    lookup_map = dict(((row[1].X, row[1].Y), row[0]) for row in sipm_df.iterrows())
    @np.vectorize
    def get_sipm_indx (X, Y):
        indx = lookup_map[(X,Y)]
        return indx
    return get_sipm_indx

types_dict_hits = OrderedDict({'event'     : np.int32  , 'time' : np.uint64, 'npeak'  : np.int32,
                               'X': np.float64, 'Y'  : np.float64    , 'Z'     : np.float64,
                               'E' : np.float64, 'E_cut'  : np.float64, 'Q'     : np.float64,
                               'Q_cut' : np.float64,  'residual': np.float64, 'reconstructed':bool})

def hits_reconstructor (*, psf_name, sipm_db, Zmax, Zmin, Xmin, Xmax, Ymax, Ymin, x_vox, y_vox, z_vox, pes_cut, lambd, spread_around, min_value_frac):
    psf = sparse.load_npz(psf_name)
    xbins = np.arange(Xmin, Xmax+x_vox, x_vox)
    ybins = np.arange(Ymin, Ymax+y_vox, y_vox)
    zbins = np.arange(Zmin, Zmax+z_vox, z_vox)
    position_to_indx = sipm_indx_getter(sipm_db)
    sol = CVXOPTQPSolver(lambd)
    def reconstruct_hits(hits_df):
        reco_hits = pd.DataFrame(columns=list(types_dict_hits.keys()))
        Z_unique_ev = np.unique(hits_df['Z'])
        #clip so that if outside the range uses the closest psf, stupid psf is 0 in last X, Y and Z bin hence -2
        Z_dig = np.clip(np.digitize(Z_unique_ev, zbins)-2, 0, len(zbins-1))
        event = hits_df.event.unique()[0]
        time = hits_df.time.unique()[0]
        E_event = hits_df.E.sum()
        #if anything goes wrong just stop and return empty df
        try:
            for iz, zuniq in enumerate(Z_unique_ev):
                #no pes_cut variable
                hits_sl = hits_df[hits_df['Z']==zuniq]
                npeak = hits_sl.npeak.unique()[0]
                E_full = hits_sl['E'].sum()
                sipm_cond = (hits_sl.Q>0)
                Q_cond = (hits_sl['Q']>pes_cut)
                if len(hits_sl[Q_cond])==0:
                    #just save slice as is
                    hits_slice = pd.DataFrame({'event':event, 'time':time, 'npeak':npeak,
                                               'X':hits_sl.X, 'Y': hits_sl.Y, 'Z':hits_sl.Z,
                                               'E':hits_sl.E, 'E_cut':0,
                                               'Q':hits_sl.Q, 'Q_cut':0,
                                               'residual':E_full/E_event, 'reconstructed':False})
                else:
                    hits_sl = hits_sl[sipm_cond].copy() #remove 0,0 hits
                    sipms_ids  = position_to_indx(hits_sl['X'].values, hits_sl['Y'].values)
                    hits_sl.loc[:, 'sipm_id'] = sipms_ids
                    Q_full = hits_sl['Q'].sum()
                    hits_slice_q = hits_sl[Q_cond]
                    
                    X_arrs = hits_slice_q['X']
                    Y_arrs = hits_slice_q['Y']
                    
                    out_full = np.zeros(len(sipm_db))
                    out_full[hits_sl.sipm_id.values] = hits_sl.Q.values/E_full
                    
                    all_values = np.concatenate([np.array(list(itertools.product(np.arange(x-spread_around, x+spread_around+np.finfo(np.float32).eps, x_vox),
                                                                                 np.arange(y-spread_around, y+spread_around+np.finfo(np.float32).eps, y_vox))))
                                                 for x, y in zip(X_arrs, Y_arrs)])
                    XY_digitz = np.vstack((np.digitize(all_values[:, 0], xbins)-1, np.digitize(all_values[:, 1], ybins)-1)).T
                    #remove values outside active volume
                    XY_digitz = XY_digitz[(XY_digitz[:, 0]>=0) & (XY_digitz[:, 1]>=0)&(XY_digitz[:, 0]<=80) & (XY_digitz[:, 1]<=80)]
                    XY_digitz = np.unique(XY_digitz, axis=0)
                    sipm_indices = hits_slice_q.sipm_id.values
                    out = np.zeros(len(sipm_db))
                    out[sipm_indices] = hits_slice_q.Q.values/E_full
                    psf_z = psf[Z_dig[iz], ...]
                    Pp = np.vstack([psf_z[i,j].todense() for i,j in zip(XY_digitz[:, 0], XY_digitz[:, 1])])
                    x0sub = np.zeros(len(XY_digitz))
                    res = sol.solve(Pp.T, out, x0sub)
                    res[res<(res.max()*min_value_frac)] = 0
                    #find output produced by reconstructed input
                    product = np.dot(Pp.T, res)
                    resid = np.square(product - out_full).sum()/np.square(out_full).sum()
                    residual = resid*E_full/E_event
                    zero_ids = np.where(product==0)[0]
                    E_out = hits_sl[hits_sl.sipm_id.isin(zero_ids)]['E'].sum()
                    Q_out = hits_sl[hits_sl.sipm_id.isin(zero_ids)]['Q'].sum()
                    E_part  = E_full - E_out
                    Q_part  = Q_full - Q_out
                    mask_pos = (res>0)
                    hits_slice = pd.DataFrame({'event':event, 'time':time, 'npeak':npeak,
                                               'X':xbins[XY_digitz[:, 0]][mask_pos], 'Y':ybins[XY_digitz[:, 1]][mask_pos], 'Z':zuniq,
                                               'E':res[mask_pos]*E_full/sum(res), 'E_cut':res[mask_pos]*E_part/sum(res),
                                               'Q':res[mask_pos]*Q_full/sum(res), 'Q_cut':res[mask_pos]*Q_part/sum(res),
                                               'residual':residual, 'reconstructed':True})
                reco_hits = reco_hits.append(hits_slice, sort=True)
            reco_hits = reco_hits.apply(lambda x : x.astype(types_dict_hits[x.name]))
            return reco_hits
        except Exception as e:
            print(e)
            return pd.DataFrame(columns=list(types_dict_hits.keys()))

    return reconstruct_hits



def hits_corrector(map_fname        : str  ,
                   apply_temp       : bool) -> Callable:
    map_fname = os.path.expandvars(map_fname)
    maps      = cof.read_maps(map_fname)
    get_coef  = cof.apply_all_correction(maps, apply_temp = apply_temp, norm_strat = cof.norm_strategy.kr)
    if maps.t_evol is not None:
        time_to_Z = cof.get_df_to_z_converter(maps)
    else:
        time_to_Z = lambda x: x
    def correct_hits(hits_df):
        hits_df = hits_df.copy()
        t = hits_df['time'].unique()[0]
        coefs = get_coef(hits_df.X.values, hits_df.Y.values, hits_df.Z.values, t)
        Ec = hits_df.E * coefs
        Ec_cut = hits_df.E_cut * coefs
        Zc = time_to_Z(hits_df.Z)
        hits_df.loc[:, 'Ec'] = Ec
        hits_df.loc[:, 'Ec_cut'] = Ec_cut
        hits_df.Z = Zc
        return hits_df
    return correct_hits

types_dict_summary = OrderedDict({'event'     : np.int32  , 'evt_energy' : np.float64, 'evt_energy_reco': np.float64,
                                  'evt_energy_reco_cut' : np.float64,  'evt_reco_nhits'  : np.int, 
                                  'evt_x_min' : np.float64, 'evt_y_min'  : np.float64, 'evt_z_min'     : np.float64,
                                  'evt_r_min' : np.float64, 'evt_x_max'  : np.float64, 'evt_y_max'     : np.float64,
                                  'evt_z_max' : np.float64, 'evt_r_max'  : np.float64, 'evt_residual'  : np.float64,
                                  'evt_E0tot' : np.float64, 'evt_E0noNN' : np.float64, 'evt_E0reco'    : np.float64})

def make_event_summary(event_number  : int ,
                       hits : pd.DataFrame 
                       ) -> pd.DataFrame:

    E0_tot = hits.E.sum()

    E0_noNN = hits[hits.Q>0].E.sum()
    E0_reco = hits[hits.reconstructed].E.sum()
    es = pd.DataFrame(columns=list(types_dict_summary.keys()))
    hits = hits[hits.reconstructed]
    
    r = np.sqrt(hits.X**2 + hits.Y**2)
    Ec_noNN = hits[hits.Q>0].Ec.sum()
    Ec_reco = hits[hits.reconstructed].Ec.sum()
    Ec_reco_cut = hits[hits.reconstructed].Ec_cut.sum()
    list_of_vars  = [event_number, Ec_noNN, Ec_reco,
                     Ec_reco_cut, sum(hits.reconstructed),
                     hits.X.min(), hits.Y.min(), hits.Z.min(), min(r),
                     hits.X.max(), hits.Y.max(), hits.Z.max(), max(r),
                     hits.residual.sum(), E0_tot, E0_noNN, E0_reco]
        
    es.loc[0] = list_of_vars
    #change dtype of columns to match type of variables
    es = es.apply(lambda x : x.astype(types_dict_summary[x.name]))
    return es

def hits_writer(h5out, compression='ZLIB4'):
    def write_hits(df):
        return df_writer(h5out              = h5out                      ,
                         df                 = df                         ,
                         compression        = compression                ,
                         group_name         = 'RECO'                  ,
                         table_name         = 'CDST'                   ,
                         descriptive_string = 'Correct hits',
                         columns_to_index   = ['event']                  )
    return write_hits

def kdst_from_df_writer(h5out, compression='ZLIB4'):
    def write_kdst(df):
        return df_writer(h5out              = h5out        ,
                         df                 = df           ,
                         compression        = compression  ,
                         group_name         = 'DST'        ,
                         table_name         = 'Events'     ,
                         descriptive_string = 'KDST Events',
                         columns_to_index   = ['event']    )
    return write_kdst


def summary_writer(h5out, compression='ZLIB4'):
    def write_summary(df):
        return df_writer(h5out              = h5out                      ,
                         df                 = df                         ,
                         compression        = compression                ,
                         group_name         = 'Summary'                  ,
                         table_name         = 'Events'                   ,
                         descriptive_string = 'Event summary information',
                         columns_to_index   = ['event']                  )
    return write_summary


@city
def reco_city(files_in, file_out, compression, event_range, print_mod,
              detector_db, run_number, map_fname, apply_temp, psf_params, reco_params):



    reconstruct_hits   = fl.map(hits_reconstructor (sipm_db = DataSiPM(detector_db, run_number), **psf_params, ** reco_params),
                                             args = 'hits',
                                             out  = 'rhits')

    correct_hits = fl.map(hits_corrector(map_fname, apply_temp),
                          args = 'rhits',
                          out  = 'chits')

    filter_events_no_hits            = fl.map(lambda x : len(x) > 0,
                                             args = 'rhits',
                                             out  = 'rhits_passed')


    hits_passed              = fl.count_filter(bool, args="rhits_passed")


    make_final_summary              = fl.map(make_event_summary,
                                             args = ('event_number', 'chits'),
                                             out  = 'event_info')

    event_count_in  = fl.spy_count()
    event_count_out = fl.spy_count()

    with tb.open_file(file_out, "w", filters=tbl.filters(compression)) as h5out:

        # Define writers...
        write_event_info = fl.sink(run_and_event_writer(h5out), args=("run_number", "event_number", "timestamp"))

        write_hits     = fl.sink( hits_writer     (h5out=h5out)                , args="chits" )
 
        write_summary         = fl.sink( summary_writer     (h5out=h5out)                , args="event_info"         )
        write_filter  = fl.sink( event_filter_writer(h5out, "high_th_select" )   , args=("event_number", "rhits_passed"))

        write_kdst_table      = fl.sink( kdst_from_df_writer(h5out)                      , args="kdst"               )

        evtnum_collect = collect()

        result = push(source = hitsdf_and_kdst_from_files(files_in),
                      pipe   = pipe(fl.slice(*event_range, close_all=True)        ,
                                    print_every(print_mod)                        ,
                                    event_count_in        .spy                    ,
                                    fl.branch(fl.fork(write_kdst_table            ,
                                                      write_event_info          )),
                                    fl.branch("event_number", evtnum_collect.sink),
                                    reconstruct_hits                              ,
                                    filter_events_no_hits                         ,
                                    fl.branch(write_filter)                       ,
                                    hits_passed.filter                            ,
                                    correct_hits                                  ,
                                    fl.branch(make_final_summary, write_summary)  ,
                                    event_count_out       .spy                    ,
                                    write_hits                                   ),
                      result = dict(events_in  =event_count_in .future,
                                    events_out =event_count_out.future,
                                    evtnum_list=evtnum_collect .future))

        if run_number <= 0:
            copy_mc_info(files_in, h5out, result.evtnum_list,
                         detector_db, run_number)

        return result
