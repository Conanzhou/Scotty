# -*- coding: utf-8 -*-
"""
Created on Fri Jun  8 10:44:34 2018

@author: VH Hall-Chen
Valerian Hongjie Hall-Chen
valerian@hall-chen.com

"""
from scotty.beam_me_up import beam_me_up
from scotty.init_bruv import get_parameters_for_Scotty
import numpy as np


# launch_freq_GHz = np.arange(34.0, 49.0, 2.0)  # Creates array [34, 36, 38, 40, 42, 44, 46, 48]
launch_freq_GHz = 40.0
shot = 12296
equil_time = 2.1
poloidal_launch_angle_Torbeam = 0.0
toroidal_launch_angle_Torbeam = 0.0

kwargs_dict = get_parameters_for_Scotty(
                              'DBS_SWIP_HL-3',
                              launch_freq_GHz = launch_freq_GHz,
                              find_B_method   = 'mdsplus', # EFITpp, UDA_saved, UDA, torbeam, mdsplus
                              equil_time      = equil_time,
                              shot            = shot,
                              user            = 'zhouyu_desktop',
                             )

kwargs_dict['mode_flag'] = 1
kwargs_dict['poloidal_launch_angle_Torbeam'] = poloidal_launch_angle_Torbeam
kwargs_dict['toroidal_launch_angle_Torbeam'] = toroidal_launch_angle_Torbeam

kwargs_dict['poloidal_flux_enter'] = 1.1**2
kwargs_dict['poloidal_flux_zero_density'] = 1.11**2
# kwargs_dict['output_path'] = os.path.dirname(os.path.abspath(__file__)) + '\\Output\\'
# kwargs_dict['density_fit_parameters'] = np.array([4.0, 1.0])
# kwargs_dict["density_fit_method"] = "quadratic"

kwargs_dict['ne_data_path'] = '/home/darkest/WorkDir/GitHub/quickfit/HL3/OUTPUT/'
kwargs_dict['input_filename_suffix'] = f'_{shot:05d}_{equil_time*1000:.0f}ms'

kwargs_dict['figure_flag'] = True

kwargs_dict['delta_R'] = -0.001
kwargs_dict['delta_Z'] = -0.001
kwargs_dict['delta_K_R'] = 0.1
kwargs_dict['delta_K_zeta'] = 0.1
kwargs_dict['delta_K_Z'] = 0.1

# kwargs_dict['quick_run'] = True

beam_me_up(**kwargs_dict)