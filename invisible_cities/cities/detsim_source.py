import numpy  as np
import tables as tb
import pandas as pd

from typing import List
from typing import Generator

from invisible_cities.io.mcinfo_io import load_mchits_df

#######################################
############### SOURCE ################
#######################################
def load_MC(files_in : List[str]) -> Generator:
    for filename in files_in:
        with tb.open_file(filename) as h5in:
            allhits = h5in.root.MC.hits.read()
            events  = np.unique(allhits["event_id"])
            for evt in events:
                hits = allhits[allhits["event_id"]==evt]

                yield dict(event_number = evt,
                           x = hits["x"],
                           y = hits["y"],
                           z = hits["z"],
                           energy = hits["energy"],
                           times = hits["time"])
