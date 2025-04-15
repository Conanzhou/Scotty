# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 12:43:55 2023

@author: VH Hall-Chen

Incorporate stuff from the old outplot.py into this

TODO: Different options for plotting.
- Plot individual graphs
- Plot groups 'all', 'basic', 'advanced', 'troubleshooting'
"""

import datatree
from scotty.plotting import (
    plot_dispersion_relation,
    plot_poloidal_beam_path,
    plot_toroidal_beam_path
    )

path = './'
dt = datatree.open_datatree(path+"scotty_output.h5", engine="h5netcdf")
# dt = datatree.open_datatree(path+"scotty_output_t4.00.h5", engine="h5netcdf")


# plot_dispersion_relation(dt['analysis'])

plot_poloidal_beam_path(dt,'test')

plot_toroidal_beam_path(dt,'test2')

dt.close()