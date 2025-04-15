# -*- coding: utf-8 -*-
"""
Created on Fri Jun  8 10:44:34 2018

@author: VH Hall-Chen
Valerian Hongjie Hall-Chen


"""

from scotty.beam_me_up import beam_me_up
import numpy as np

shot = 12296
equil_time = 2.1

kwargs_dict = {
    "poloidal_launch_angle_Torbeam": 0.0,
    "toroidal_launch_angle_Torbeam": 0.0,
    "launch_freq_GHz": 40.0,
    "mode_flag": 1,
    "launch_beam_width": 0.04,
    "launch_beam_curvature": -0.25,
    "launch_position": np.array([2.75, 0.0, -0.15]),
    # "density_fit_parameters": np.array([8.0, 1.0]),
    "delta_R": -0.00001,
    "delta_Z": 0.00001,
    # "density_fit_method": "quadratic",
    "find_B_method": "analytical",
    "Psi_BC_flag": True,
    "figure_flag": True,
    "vacuum_propagation_flag": True,
    "vacuumLaunch_flag": True,
    "poloidal_flux_enter": 1.0,
    "poloidal_flux_zero_density": 1.0,
    "B_T_axis": 1.3,
    "B_p_a": 0.1,
    "R_axis": 1.78,
    "minor_radius_a": 0.6,
}

kwargs_dict['ne_data_path'] = '/home/darkest/WorkDir/GitHub/quickfit/HL3/OUTPUT/'
kwargs_dict['input_filename_suffix'] = f'_{shot:05d}_{equil_time*1000:.0f}ms'

beam_me_up(**kwargs_dict)
