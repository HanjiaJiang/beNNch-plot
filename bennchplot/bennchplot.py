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

"""
Class for benchmarking plots
"""
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
from matplotlib import gridspec
import numpy as np
import yaml
import os
try:
    from . import plot_params as pp
except ImportError:
    import plot_params as pp


class Plot():
    """
    Class organizing benchmarking plots.

    Attributes
    ----------
    x_axis : str or list
        variable to be plotted on x-axis
    x_ticks : str, optional

    data_file : str, optional
        path to data
    matplotlib_params : dict, optional
        parameters passed to matplotlib
    color_params : dict, optional
        unique colors for variables
    additional_params : dict, optional
        additional parameters used for plotting
    label_params : dict, optional
        labels used when plotting
    time_scaling : int, optional
        scaling parameter for simulation time
   """

    def __init__(self, x_axis,
                 x_ticks='data',
                 data_file='/path/to/data',
                 matplotlib_params=pp.matplotlib_params,
                 color_params=pp.color_params,
                 additional_params=pp.additional_params,
                 label_params=pp.label_params,
                 time_scaling=1,
                 ctrl_file='/path/to/control'):

        self.x_axis = x_axis
        self.x_ticks = x_ticks
        self.matplotlib_params = matplotlib_params
        self.additional_params = additional_params
        self.color_params = color_params
        self.label_params = label_params
        self.time_scaling = time_scaling
        self.df_data = None
        self.df_ctrl = None

        self.load_data(data_file)
        self.load_data(ctrl_file, control=True)
        self.compute_derived_quantities()
        self.compute_derived_quantities(control=True)

    def load_data(self, data_file, control=False):
        """
        Load data to dataframe, to be used later when plotting.

        Group the data by specified operations.

        Attributes
        ----------
        data_file : str
            data file to be loaded and later plotted

        Raises
        ------
        ValueError
        """
        print(f'Loading {data_file} ...')
        try:
            df = pd.read_csv(data_file, delimiter=',')
        except FileNotFoundError:
            print('File could not be found')
            quit()

        for py_timer in ['py_time_create', 'py_time_connect']:
            if py_timer not in df:
                df[py_timer] = np.nan
                raise ValueError('Warning! Python timers are not found. ' +
                                 'Construction time measurements will not ' +
                                 'be accurate.')

        dict_ = {'num_nodes': 'first',
                 'threads_per_task': 'first',
                 'tasks_per_node': 'first',
                 'model_time_sim': 'first',
                 'wall_time_create': ['mean', 'std'],
                 'wall_time_connect': ['mean', 'std'],
                 'wall_time_sim': ['mean', 'std'],
                 'wall_time_phase_collocate': ['mean', 'std'],
                 'wall_time_phase_communicate': ['mean', 'std'],
                 'wall_time_phase_deliver': ['mean', 'std'],
                 'wall_time_phase_update': ['mean', 'std'],
                 'wall_time_communicate_target_data': ['mean', 'std'],
                 'wall_time_gather_spike_data': ['mean', 'std'],
                 'wall_time_gather_target_data': ['mean', 'std'],
                 'wall_time_communicate_prepare': ['mean', 'std'],
                 'py_time_create': ['mean', 'std'],
                 'py_time_connect': ['mean', 'std'],
                 'network_size': 'first',
                 'base_memory': ['mean', 'std'],
                 'network_memory': ['mean', 'std'],
                 'init_memory': ['mean', 'std'],
                 'total_memory': ['mean', 'std'],
                 'num_connections': ['mean', 'std'],
                 'local_spike_counter': ['mean', 'std'],
                 }

        col = ['num_nodes', 'threads_per_task', 'tasks_per_node',
               'model_time_sim', 'wall_time_create',
               'wall_time_create_std', 'wall_time_connect',
               'wall_time_connect_std', 'wall_time_sim',
               'wall_time_sim_std', 'wall_time_phase_collocate',
               'wall_time_phase_collocate_std', 'wall_time_phase_communicate',
               'wall_time_phase_communicate_std', 'wall_time_phase_deliver',
               'wall_time_phase_deliver_std', 'wall_time_phase_update',
               'wall_time_phase_update_std',
               'wall_time_communicate_target_data',
               'wall_time_communicate_target_data_std',
               'wall_time_gather_spike_data',
               'wall_time_gather_spike_data_std',
               'wall_time_gather_target_data',
               'wall_time_gather_target_data_std',
               'wall_time_communicate_prepare',
               'wall_time_communicate_prepare_std',
               'py_time_create', 'py_time_create_std',
               'py_time_connect', 'py_time_connect_std',
               'network_size',
               'base_memory', 'base_memory_std',
               'network_memory', 'network_memory_std',
               'init_memory', 'init_memory_std',
               'total_memory', 'total_memory_std',
               'num_connections', 'num_connections_std',
               'local_spike_counter', 'local_spike_counter_std',]

        df = df.drop('rng_seed', axis=1).groupby(
            ['num_nodes',
             'threads_per_task',
             'tasks_per_node',
             'model_time_sim'], as_index=False).agg(dict_)
        df.columns = col
        if control:
            self.df_ctrl = df.copy()
        else:
            self.df_data = df.copy()

    def compute_derived_quantities(self, control=False):
        """
        Do computations to get parameters needed for plotting.
        """
        if control:
            df = self.df_ctrl
        else:
            df = self.df_data
        df['num_nvp'] = (
            df['threads_per_task'] * df['tasks_per_node']
        )
        df['model_time_sim'] /= self.time_scaling
        df['wall_time_create+wall_time_connect'] = (
            df['py_time_create'] + df['py_time_connect'])
        df['wall_time_create+wall_time_connect_std'] = (
            np.sqrt((df['wall_time_create_std']**2 +
                     df['wall_time_connect_std']**2)))
        df['sim_factor'] = (df['wall_time_sim'] /
                                 df['model_time_sim'])
        df['sim_factor_std'] = (df['wall_time_sim_std'] /
                                     df['model_time_sim'])
        df['wall_time_phase_total'] = (
            df['wall_time_phase_update'] +
            df['wall_time_phase_communicate'] +
            df['wall_time_phase_deliver'] +
            df['wall_time_phase_collocate'])
        df['wall_time_phase_total_std'] = \
            np.sqrt(
            df['wall_time_phase_update_std']**2 +
            df['wall_time_phase_communicate_std']**2 +
            df['wall_time_phase_deliver_std']**2 +
            df['wall_time_phase_collocate_std']**2
        )
        df['phase_total_factor'] = (
            df['wall_time_phase_total'] /
            df['model_time_sim'])
        df['phase_total_factor_std'] = (
            df['wall_time_phase_total_std'] /
            df['model_time_sim'])

        for phase in ['update', 'communicate', 'deliver', 'collocate']:
            df['phase_' + phase + '_factor'] = (
                df['wall_time_phase_' + phase] /
                df['model_time_sim'])

#            df['phase_' + phase + '_factor' + '_std'] = (
#                df['wall_time_phase_' + phase + '_std'] /
#                df['model_time_sim'])

            df['frac_phase_' + phase] = (
                100 * df['wall_time_phase_' + phase] /
                df['wall_time_sim'])

#            df['frac_phase_' + phase + '_std'] = (
#                100 * df['wall_time_phase_' + phase + '_std'] /
#                df['wall_time_phase_total'])
        # signal transmission = communicate + deliver + collocate
        df['phase_signal_transmission_factor'] = (
            df['phase_communicate_factor'] +
            df['phase_deliver_factor'] +
            df['phase_collocate_factor'])
#        df['phase_signal_transmission_factor_std'] = \
#            np.sqrt(
#            df['phase_communicate_factor_std']**2 +
#            df['phase_deliver_factor_std']**2 +
#            df['phase_collocate_factor_std']**2
#        )
        df['frac_phase_signal_transmission'] = (
            df['frac_phase_communicate'] +
            df['frac_phase_deliver'] +
            df['frac_phase_collocate'])
#        df['frac_phase_signal_transmission_std'] = \
#            np.sqrt(
#            df['frac_phase_communicate_std']**2 +
#            df['frac_phase_deliver_std']**2 +
#            df['frac_phase_collocate_std']**2
#        )
        # others = the rest
        df['phase_others_factor'] = (df['wall_time_sim'] - df['wall_time_phase_total'])/df['model_time_sim']
        df['frac_phase_others'] = (100 - (df['frac_phase_signal_transmission'] + df['frac_phase_update']))

        df['total_memory_per_node'] = (df['total_memory'] /
                                            df['num_nodes'])
        df['total_memory_per_node_std'] = (df['total_memory_std'] /
                                                df['num_nodes'])
        df['total_spike_count_per_s'] = (df['local_spike_counter'] / df['model_time_sim'])
        df['total_spike_count_per_s_std'] = (df['local_spike_counter_std'] / df['model_time_sim'])

    def plot_fractions(self, axis, fill_variables,
                       interpolate=False, step=None, log=False, alpha=1.,
                       error=False, control=False, line=False, subject=None, ylims=None):
        """
        Fill area between curves.

        axis : Matplotlib axes object
        fill_variables : list
            variables (e.g. timers) to be plotted as fill  between graph and
            x axis
        interpolate : bool, default
            whether to interpolate between the curves
        step : {'pre', 'post', 'mid'}, optional
            should the filling be a step function
        log : bool, default
            whether the x-axes should have logarithmic scale
        alpha, int, default
            alpha value of fill_between plot
        error : bool
            whether plot should have error bars
        """
        if control:
            df = self.df_ctrl
        else:
            df = self.df_data

        fill_height = 0
        for i, fill in enumerate(fill_variables):
            main_label = subject if isinstance(subject, str) else None
            main_label = main_label if fill == fill_variables[-1] else None
            line_color = 'gray' if control else 'k'
            if control:
                pass
            else:
                frac_label = self.label_params[fill]
                if isinstance(subject, str):
                    frac_label += f' ({subject})'
                axis.fill_between(np.squeeze(df[self.x_axis]),
                                  fill_height,
                                  np.squeeze(df[fill]) + fill_height,
                                  label=frac_label,
                                  facecolor=self.color_params[fill],
                                  interpolate=interpolate,
                                  step=step,
                                  alpha=alpha,
                                  linewidth=0.5,
                                  edgecolor='#444444')
            if error:
                axis.errorbar(np.squeeze(df[self.x_axis]),
                              np.squeeze(df[fill]) + fill_height,
                              yerr=np.squeeze(df[fill + '_std']),
                              capsize=3,
                              capthick=1,
                              color=line_color,
                              fmt='none',
                              label=main_label
                              )
            fill_height += df[fill].to_numpy()

        if self.x_ticks == 'data':
            axis.set_xticks(np.squeeze(df[self.x_axis]))
        else:
            axis.set_xticks(self.x_ticks)

        if isinstance(ylims, tuple):
            axis.set_ylim(ylims)

        if log:
            axis.set_xscale('log')
            axis.tick_params(bottom=False, which='minor')
            axis.get_xaxis().set_major_formatter(
                matplotlib.ticker.ScalarFormatter())

    def plot_main(self, quantities, axis, log=(False, False),
                  error_only=False, fmt='none', control=False, subject=None, line_color=None, ylims=None):
        """
        Main plotting function.

        Attributes
        ----------
        quantities : list
            list with plotting quantities
        axis : axis object
            axis object used when plotting
        log : tuple of bools, default
            whether x and y axis should have logarithmic scale
        error : bool, default
            whether or not to plot error bars
        fmt : string
            matplotlib format string (fmt) for defining line style
        """
        if control:
            df = self.df_ctrl
        else:
            df = self.df_data

        for y in quantities:
            line_style = ':' if control else '-'
            line_color = self.color_params[y] if line_color is None else line_color
            label = subject if isinstance(subject, str) else self.label_params[y]
            if not error_only:
                axis.plot(df[self.x_axis],
                          df[y],
                          marker=None,
                          color=line_color,
                          linewidth=1,
                          linestyle=line_style)
            axis.errorbar(
                df[self.x_axis].values,
                df[y].values,
                yerr=df[y + '_std'].values,
                marker=None,
                capsize=3,
                capthick=1,
                label=label,
                color=line_color,
                fmt=fmt)

        # if self.x_ticks == 'data':
        #    axis.set_xticks(df[self.x_axis].values)
        # else:
        #    axis.set_xticks(self.x_ticks)

        if isinstance(ylims, tuple):
            axis.set_ylim(ylims)

        if log[0]:
            axis.set_xscale('log')
        if log[1]:
            axis.tick_params(bottom=False, which='minor')
            axis.set_yscale('log')

    def merge_legends(self, ax1, ax2):
        """
        Merge legends from two axes, display them in the first

        Attributes
        ----------
        ax1 : axes object
            first axis
        ax2 : axes object
            second axis
        """
        handles, labels = [(a + b) for a, b in zip(
            ax2.get_legend_handles_labels(),
            ax1.get_legend_handles_labels())]
        ax1.legend(handles, labels, loc='upper right')

    def simple_axis(self, ax):
        """
        Remove top and right spines.

        Attributes
        ----------
        ax : axes object
            axes object for which to adjust spines
        """
        # Hide the right and top spines
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        # Only show ticks on the left and bottom spines
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')
