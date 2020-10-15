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

#from .. core.system_of_units import *
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

def pmaps_df_from_files(paths: List[str]) -> Iterator[Dict[str,Union[pd.DataFrame, pd.DataFrame, MCInfo, int, float]]]:
    for path in paths:
        with tb.open_file(path, "r") as h5in:
            try:
                run_number  = get_run_number(h5in)
                event_info  = get_event_info(h5in)
                S2_table    = h5in.root.PMAPS.S2
                S1_table    = h5in.root.PMAPS.S1
                S2Si_table  = h5in.root.PMAPS.S2Si
            except (tb.exceptions.NoSuchNodeError, IndexError):
                continue

            #check_lengths(event_info, hits_df.event.unique())

            for evtinfo in event_info:
                event_number, timestamp = evtinfo.fetch_all_fields()
                s2 = pd.DataFrame.from_records(S2_table.read_where( 'event == {}'.format(event_number)))
                s1 = pd.DataFrame.from_records(S1_table.read_where( 'event == {}'.format(event_number)))
                s2sipm = pd.DataFrame.from_records(S2Si_table.read_where( 'event == {}'.format(event_number)))

                if len(s2) & len(s1) & len(s2sipm):
                    yield dict(s2 = s2, s1 = s1, s2sipm = s2sipm,
                               run_number = run_number,
                               event_number = event_number,
                               timestamp = timestamp)
                else:
                    continue

def get_df_from_pmaps_(*,drift_velocity, stride):
    def get_df_from_pmaps(s1, s2, s2sipm):
        """
        This returns dataframe after rebinning with relevant info
        """
        s2 = s2.copy()
        s2sipm = s2sipm.copy()
        s2sipm.ene = s2sipm.ene.astype(np.float64)
        t0 = s1.time[s1.ene.argmax()]
        nsipms = s2sipm.groupby('peak').nsipm.nunique().values #list of 
        time_len = s2.groupby('peak').time.count().values
        s2.loc[:, 'rebin_indx']=np.hstack([np.repeat(np.arange(0, peak_lngt//stride+1), stride)
                                           [:peak_lngt] for peak_lngt in time_len])
        times = s2.groupby('peak').time.apply(lambda x : np.tile(x-t0, nsipms[x.name])).explode().astype(np.float).reset_index(drop=True)
        rebin_indx = s2.groupby('peak').rebin_indx.apply(lambda x : np.tile(x, nsipms[x.name])).explode().astype(np.int).reset_index(drop=True)
        pmt_energ = s2.groupby('peak').ene.apply(lambda x : np.tile(x, nsipms[x.name])).explode().astype(np.float).reset_index(drop=True)
        s2sipm.loc[:, 'dt'] = times
        s2sipm.loc[:, 'rebin_indx'] = rebin_indx
        s2sipm.loc[:, 'pmt_ene']=pmt_energ
        s2sipm_pr = s2sipm.groupby(['peak', 'rebin_indx', 'nsipm']).agg({'ene':np.sum,'dt':np.mean, 'pmt_ene':np.sum}).reset_index()
        s2sipm_pr.drop('rebin_indx', axis=1, inplace=True)
        rebined_df = s2sipm_pr.groupby(['dt']).apply(lambda row: row['ene']*(row['pmt_ene'].mean())/(row['ene'].sum()))
        s2sipm_pr.loc[rebined_df.index.get_level_values(1), 'pmt_ene_ps']=rebined_df.values
        #s2sipm_pr.drop(labels='pmt_ene', axis=1, inplace=True)
        s2sipm_pr = s2sipm_pr.rename(columns={'ene':'Q', 'pmt_ene_ps':'E', 'pmt_ene':'sliceE'})
        s2sipm_pr.loc[:, 'Z'] = s2sipm_pr.dt*drift_velocity
        s2sipm_pr_pos  = s2sipm_pr[np.isfinite(s2sipm_pr.E)&(s2sipm_pr.Q>0)]
        s2sipm_pr_neg = s2sipm_pr[~np.isfinite(s2sipm_pr.E)].drop_duplicates('Z')
        s2sipm_pr = pd.concat([s2sipm_pr_pos, s2sipm_pr_neg]).sort_values(['Z', 'nsipm']).reset_index(drop=True)
        #s2sipm_pr[s2sipm_pr.Q<0] = s2sipm_pr.reset_index(drop=True)#[s2sipm_pr.Q>0]
        total_energy = s2.ene.sum()
        return s2sipm_pr, total_energy
    return get_df_from_pmaps

types_dict_hits = OrderedDict({'event'     : np.int32  , 'time' : np.uint64, 'npeak'  : np.int32,
                               'X': np.float64, 'Y'  : np.float64    , 'Z'     : np.float64,
                               'E' : np.float64, 'E_cut'  : np.float64, 'Q'     : np.float64,
                               'Q_cut' : np.float64,  'residual': np.float64, 'reconstructed':bool})

def hits_reconstructor (*, psf_name, sipm_db, Zmax, Zmin, Xmin, Xmax, Ymax, Ymin, x_vox, y_vox, z_vox, pes_cut, lambd, spread_around, min_reco_charge):
    psf = sparse.load_npz(psf_name)
    xbins = np.arange(Xmin, Xmax+x_vox, x_vox)
    ybins = np.arange(Ymin, Ymax+y_vox, y_vox)
    zbins = np.arange(Zmin, Zmax+z_vox, z_vox)
    sol = CVXOPTQPSolver(lambd)
    def reconstruct_hits(phits_df, total_energy, event, time):
        reco_hits = pd.DataFrame(columns=list(types_dict_hits.keys()))
        Z_unique_ev = np.unique(phits_df['Z'])
        #clip so that if outside the range uses the closest psf, stupid psf is 0 in last X, Y and Z bin hence -2
        Z_dig = np.clip(np.digitize(Z_unique_ev, zbins)-2, 0, len(zbins-1))
        E_event = total_energy
        #if anything goes wrong just stop and return empty df
        try:
            for iz, zuniq in enumerate(Z_unique_ev):
                hits_sl = phits_df[phits_df['Z']==zuniq]
                npeak = hits_sl.peak.unique()[0]
                E_full = hits_sl['E'].sum()
                Q_cond = (hits_sl['Q']>pes_cut)
                if len(hits_sl[Q_cond])==0:
                    #just save slice as is
                    hits_slice = pd.DataFrame({'event':event, 'time':time, 'npeak':npeak,
                                               'X':sipm_db.X.values[hits_sl.nsipm.values], 
                                               'Y':sipm_db.X.values[hits_sl.nsipm.values], 'Z':hits_sl.Z,
                                               'E':hits_sl.E, 'E_cut':0,
                                               'Q':hits_sl.Q, 'Q_cut':0,
                                               'residual':E_full/E_event, 'reconstructed':False})
                else:
                    Q_full = hits_sl['Q'].sum()
                    hits_slice_q = hits_sl[Q_cond]
                    X_arrs = sipm_db.X.values[hits_slice_q.nsipm.values]
                    Y_arrs = sipm_db.Y.values[hits_slice_q.nsipm.values]
                    
                    
                    out_full = np.zeros(len(sipm_db))
                    out_full[hits_sl.nsipm.values] = hits_sl.Q.values/E_full
                    
                    all_values = np.concatenate([np.array(list(itertools.product(np.arange(x-spread_around, x+spread_around+np.finfo(np.float32).eps, x_vox),
                                                                                 np.arange(y-spread_around, y+spread_around+np.finfo(np.float32).eps, y_vox))))
                                                 for x, y in zip(X_arrs, Y_arrs)])
                    XY_digitz = np.vstack((np.digitize(all_values[:, 0], xbins)-1, np.digitize(all_values[:, 1], ybins)-1)).T
                    #remove values outside active volume
                    XY_digitz = XY_digitz[(XY_digitz[:, 0]>=0) & (XY_digitz[:, 1]>=0)&(XY_digitz[:, 0]<=80) & (XY_digitz[:, 1]<=80)]
                    XY_digitz = np.unique(XY_digitz, axis=0)
                    sipm_indices = hits_slice_q.nsipm.values
                    out = np.zeros(len(sipm_db))
                    out[sipm_indices] = hits_slice_q.Q.values/E_full
                    psf_z = psf[Z_dig[iz], ...]
                    Pp = np.vstack([psf_z[i,j].todense() for i,j in zip(XY_digitz[:, 0], XY_digitz[:, 1])])
                    x0sub = np.zeros(len(XY_digitz))
                    res = sol.solve(Pp.T, out, x0sub)
                    #clip to 0 all hits with charge smaller than min_charge
                    charge_fr_indx  = Q_full/sum(res)
                    res[res*charge_fr_indx<min_reco_charge] = 0
                    #find output produced by reconstructed input
                    product = np.dot(Pp.T, res)
                    resid = np.square(product - out_full).sum()/np.square(out_full).sum()
                    residual = resid*E_full/E_event
                    zero_ids = np.where(product==0)[0]
                    E_out = hits_sl[hits_sl.nsipm.isin(zero_ids)]['E'].sum()
                    Q_out = hits_sl[hits_sl.nsipm.isin(zero_ids)]['Q'].sum()
                    E_part  = E_full - E_out
                    Q_part  = Q_full - Q_out
                    mask_pos = (res>0)
                    hits_slice = pd.DataFrame({'event':event, 'time':time, 'npeak':npeak,
                                               'X':xbins[XY_digitz[:, 0]][mask_pos], 'Y':ybins[XY_digitz[:, 1]][mask_pos], 'Z':zuniq,
                                               'E':res[mask_pos]*E_full/sum(res), 'E_cut':res[mask_pos]*E_part/sum(res),
                                               'Q':res[mask_pos]*Q_full/sum(res), 'Q_cut':res[mask_pos]*Q_part/sum(res),
                                               'residual':residual, 'reconstructed':True})
                #print(hits_slice)
                reco_hits = reco_hits.append(hits_slice, sort=True, ignore_index=True)
            reco_hits = reco_hits.apply(lambda x : x.astype(types_dict_hits[x.name]))
            return reco_hits
        except Exception as e:
            print(e)
            return pd.DataFrame(columns=list(types_dict_hits.keys()))
    return reconstruct_hits


def hits_writer(h5out, compression='ZLIB4'):
    def write_hits(df):
        return df_writer(h5out              = h5out                      ,
                         df                 = df                         ,
                         compression        = compression                ,
                         group_name         = 'RECO'                  ,
                         table_name         = 'RDST'                   ,
                         descriptive_string = 'Reconstructed hits',
                         columns_to_index   = ['event']                  )
    return write_hits


@city
def reco_city(files_in, file_out, compression, event_range, print_mod,
              detector_db, run_number, drift_velocity, stride, psf_params, reco_params):


    get_hits = fl.map(get_df_from_pmaps_(drift_velocity=drift_velocity, stride=stride),
                      args = ('s1',  's2', 's2sipm'),
                      out = ('phits', 'energy'))

    reconstruct_hits   = fl.map(hits_reconstructor (sipm_db = DataSiPM(detector_db, run_number), **psf_params, ** reco_params),
                                args = ('phits', 'energy', 'event_number', 'timestamp'),
                                out  = 'rhits')


    filter_events_no_hits            = fl.map(lambda x : len(x) > 0,
                                             args = 'rhits',
                                             out  = 'rhits_passed')

    hits_passed              = fl.count_filter(bool, args="rhits_passed")

    event_count_in  = fl.spy_count()
    event_count_out = fl.spy_count()

    with tb.open_file(file_out, "w", filters=tbl.filters(compression)) as h5out:

        # Define writers...
        write_event_info = fl.sink(run_and_event_writer(h5out), args=("run_number", "event_number", "timestamp"))

        write_hits     = fl.sink( hits_writer     (h5out=h5out)                , args="rhits" )
        write_filter  = fl.sink( event_filter_writer(h5out, "high_th_select" )   , args=("event_number", "rhits_passed"))
        evtnum_collect = collect()

        result = push(source = pmaps_df_from_files(files_in),
                      pipe   = pipe(fl.slice(*event_range, close_all=True)        ,
                                    print_every(print_mod)                        ,
                                    event_count_in        .spy                    ,
                                    fl.branch(write_event_info                   ),
                                    fl.branch("event_number", evtnum_collect.sink),

                                    get_hits                                      ,
                                    reconstruct_hits                              ,
                                    filter_events_no_hits                         ,
                                    fl.branch(write_filter)                       ,
                                    hits_passed.filter                            ,
                                    event_count_out       .spy                    ,
                                    write_hits                                   ),
                      result = dict(events_in  =event_count_in .future,
                                    events_out =event_count_out.future,
                                    evtnum_list=evtnum_collect .future))

        if run_number <= 0:
            copy_mc_info(files_in, h5out, result.evtnum_list,
                         detector_db, run_number)

        return result
