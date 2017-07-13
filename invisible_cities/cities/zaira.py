import sys
import time

import numpy  as np
import tables as tb
import pandas as pd

from .  base_cities         import City
from .  base_cities         import MapCity
from .. core.fit_functions  import in_range
from .. core.configure      import configure
from .. reco.dst_functions  import load_dst
from .. io.kdst_io           import xy_writer

from .  diomira     import Diomira


class Zaira(MapCity):

    go = Diomira.go

    def __init__(self, **kwds):
        super().__init__(**kwds)
        conf = self.conf

        fiducial_z, fiducial_e = conf.fiducial_z, conf.fiducial_e

        self._fiducial_z = ((self.det_geo.ZMIN[0], self.det_geo.ZMAX[0])
                             if fiducial_z is None
                             else fiducial_z)
        self._fiducial_e = (0, np.inf) if fiducial_e is None else fiducial_e

        self._dst_group  = conf.dst_group
        self._dst_node   = conf.dst_node

    def run(self):
        dsts = [load_dst(input_file, self._dst_group, self._dst_node)
                for input_file in self.input_files]

        # Correct each dataset with the corresponding lifetime
        for dst, correct in zip(dsts, self._lifetime_corrections):
            dst.S2e *= correct(dst.Z.values).value

        # Join datasets
        dst = pd.concat(dsts)
        dst = dst[in_range(dst.S2e.values, *self._fiducial_e)]
        dst = dst[in_range(dst.Z  .values, *self._fiducial_z)]
        dst = dst[in_range(dst.X  .values, *self._xrange    )]
        dst = dst[in_range(dst.Y  .values, *self._yrange    )]

        # Compute corrections and stats
        xycorr = self.xy_correction(dst.X.values, dst.Y.values, dst.S2e.values)
        nevt   = self.xy_statistics(dst.X.values, dst.Y.values)[0]

        with tb.open_file(self.output_file, 'w') as h5out:
            write_xy = xy_writer(h5out)
            write_xy(*xycorr._xs, xycorr._fs, xycorr._us, nevt)

        return len(dst)