"""
beNNch-plot - standardized plotting routines for performance benchmarks.
Copyright (C) 2021 Forschungszentrum Juelich GmbH, INM-6

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version.
This program is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with
this program. If not, see <https://www.gnu.org/licenses/>.

SPDX-License-Identifier: GPL-3.0-or-later
"""
import numpy as np
import bennchplot as bp
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec


"""
define what to plot:
- data_file:
    Path to .csv file containing benchmarking measurements.
- x_axis:
    Giving a list of strings corresponding to the main scaling
    variable, typically 'num_nodes' or 'num_nvp'.
- time_scaling:
    Quotient between unit time of timing measurement and
    simulation. Usually, the former is given in s while
    the latter is given in ms. 
"""
args = {
    'data_file': '45011f6d-c3c2-4f2c-b884-af04e9edc5b9.csv',
    'x_axis': ['num_nodes'],
    'time_scaling': 1e3
}


# Instantiate class
B = bp.Plot(**args)

# Figure layout
widths = [1, 1]
heights = [3, 1]
fig = plt.figure(figsize=(12, 6), constrained_layout=True)
spec = gridspec.GridSpec(ncols=2, nrows=2, figure=fig,
                         width_ratios=widths, height_ratios=heights)

ax1 = fig.add_subplot(spec[0, 0])
ax1_twin = ax1.twiny()
ax1a = fig.add_subplot(spec[1, 0])
ax2 = fig.add_subplot(spec[0, 1])
ax2_twin = ax2.twiny()
ax3 = fig.add_subplot(spec[1, 1])

# Add plots
B.plot_fractions(axis=ax1,
                 fill_variables=['wall_time_create+wall_time_connect',
                                 'wall_time_sim'],
                 interpolate=True,
                 step=None,
                 error=True)
B.plot_main(quantities=['total_spike_count_per_s'], axis=ax1a, error=False)
B.plot_main(quantities=['total_spike_count_per_s'], axis=ax1a, error=True)
B.plot_main(quantities=['sim_factor'], axis=ax2, error=True)
B.plot_fractions(axis=ax2,
                 fill_variables=[
                     'phase_update_factor',
                     'phase_collocate_factor',
                     'phase_communicate_factor',
                     'phase_deliver_factor'
                 ])
B.plot_fractions(axis=ax3,
                 fill_variables=[
                     'frac_phase_communicate',
                     'frac_phase_update',
                     'frac_phase_deliver'])

# Set labels, limits etc.
ax1a.set_ylim(bottom=0)
ax2.set_ylim(0, 300)
#B.simple_axis(ax1)
#B.simple_axis(ax2)
#B.simple_axis(ax1a)

ax1.set_ylabel(r'$T_{\mathrm{wall}}$ [s] for $T_{\mathrm{model}} =$'
               + f'{np.unique(B.df.model_time_sim.values)[0]} s')
ax1a.set_xlabel('Number of nodes')
ax1a.set_ylabel('Total spikes/s')
ax2.set_ylabel(r'real-time factor $T_{\mathrm{wall}}/$'
               r'$T_{\mathrm{model}}$')
ax3.set_xlabel('Number of nodes')
ax3.set_ylabel(r'relative $T_{\mathrm{wall}}$ [%]')

ax1.legend()
ax2.legend()

ax1_twin.set_xticks(ax1.get_xticks())
ax1_twin.set_xticklabels(['20000', '40000', '60000', '80000', '100000', '1200000'])
ax1_twin.set_xlabel('Number of cells')
ax2_twin.set_xticks(ax2.get_xticks())
ax2_twin.set_xticklabels(['20000', '40000', '60000', '80000', '100000', '1200000'])
ax2_twin.set_xlabel('Number of cells')
ax1a.set_xticks(ax1.get_xticks())
ax1.set_xticklabels([])
ax2.set_xticklabels([])

ax1_twin.set_xlim(ax1.get_xlim())
ax2_twin.set_xlim(ax2.get_xlim())

# Save figure
plt.savefig('scaling.pdf')
