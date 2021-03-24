#  LLEPE: Liquid-Liquid Equilibrium Parameter Estimator
#  Copyright (C) 2020, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See LICENSE for more details.

import llepe
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib
import re


def set_size(w, h, ax=None):
    """ w, h: width, height in inches """
    if not ax:
        ax = plt.gca()
    left = ax.figure.subplotpars.left
    right = ax.figure.subplotpars.right
    top = ax.figure.subplotpars.top
    bottom = ax.figure.subplotpars.bottom
    fig_width = float(w) / (right - left)
    fig_height = float(h) / (top - bottom)
    ax.figure.set_size_inches(fig_width, fig_height)


font = {'family': 'sans serif',
        'size': 24}
matplotlib.rc('font', **font)
matplotlib.rc('xtick', labelsize=18)
matplotlib.rc('ytick', labelsize=18)
matplotlib.rcParams['lines.linewidth'] = 4
matplotlib.rcParams['lines.markersize'] = 10


def ext_to_complex(h0, custom_obj_dict, mini_species):
    linear_params = custom_obj_dict['lin_param_df']
    row = linear_params[linear_params['species'] == mini_species]
    return row['slope'].values[0] * h0[0] + row['intercept'].values[0]


def mod_lin_param_df(lp_df, input_val, mini_species, mini_lin_param):
    new_lp_df = lp_df.copy()
    index = new_lp_df.index[new_lp_df['species'] == mini_species].tolist()[0]
    new_lp_df.at[index, mini_lin_param] = input_val
    return new_lp_df


info_df = pd.read_csv('outputs/iterative_fitter_output4.csv')
test_row = -1
pitzer_params_filename = "../../data/jsons/min_h0_pitzer_params.txt"
with open(pitzer_params_filename) as file:
    pitzer_params_dict = json.load(file)
pitzer_params_df = pd.DataFrame(pitzer_params_dict)
species_list = 'Nd,Pr,Ce,La,Dy,Sm,Y'.split(',')
pitzer_param_list = ['beta0', 'beta1']
labeled_data = pd.read_csv("../../data/csvs/"
                           "PC88A_HCL_NdPrCeLaDySmY.csv")
labeled_data = labeled_data.sort_values(['Feed Pr[M]', 'Feed Ce[M]'],
                                        ascending=True)
exp_data = labeled_data.drop(labeled_data.columns[0], axis=1)
xml_file = "PC88A_HCL_NdPrCeLaDySmY_w_pitzer.xml"
lin_param_df = pd.read_csv("../../data/csvs"
                           "/zeroes_removed_min_h0_pitzer_lin_params.csv")
estimator_params = {'exp_data': exp_data,
                    'phases_xml_filename': xml_file,
                    'phase_names': ['HCl_electrolyte', 'PC88A_liquid'],
                    'aq_solvent_name': 'H2O(L)',
                    'extractant_name': '(HA)2(org)',
                    'diluant_name': 'dodecane',
                    'complex_names': ['{0}(H(A)2)3(org)'.format(species)
                                      for species in species_list],
                    'extracted_species_ion_names': ['{0}+++'.format(species)
                                                    for species in
                                                    species_list],
                    'aq_solvent_rho': 1000.0,
                    'extractant_rho': 960.0,
                    'diluant_rho': 750.0,
                    'temp_xml_file_path': 'outputs/temp.xml',
                    'objective_function': llepe.lmse_perturbed_obj
                    }
dependant_params_dict = {}
for species, complex_name in zip(species_list,
                                 estimator_params['complex_names']):
    inner_dict = {'upper_element_name': 'species',
                  'upper_attrib_name': 'name',
                  'upper_attrib_value': complex_name,
                  'lower_element_name': 'h0',
                  'lower_attrib_name': None,
                  'lower_attrib_value': None,
                  'input_format': '{0}',
                  'function': ext_to_complex,
                  'kwargs': {"mini_species": species},
                  'independent_params': '(HA)2(org)_h0'}
    dependant_params_dict['{0}_h0'.format(complex_name)] = inner_dict
info_dict = {'(HA)2(org)_h0': {'upper_element_name': 'species',
                               'upper_attrib_name': 'name',
                               'upper_attrib_value': '(HA)2(org)',
                               'lower_element_name': 'h0',
                               'lower_attrib_name': None,
                               'lower_attrib_value': None,
                               'input_format': '{0}',
                               'input_value':
                                   info_df.iloc[test_row, :]['best_ext_h0']}}
for species in species_list:
    for pitzer_param in pitzer_param_list:
        pitzer_str = "{0}_{1}".format(species, pitzer_param)
        value = info_df.iloc[test_row, :][pitzer_str]
        pitzer_params_dict[pitzer_str]['input_value'] = value
    lin_str = "{0}_slope".format(species)
    inner_dict = {'custom_object_name': 'lin_param_df',
                  'function': mod_lin_param_df,
                  'kwargs': {'mini_species': species,
                             'mini_lin_param': 'slope'},
                  'input_value': 3
                  }
    info_dict[lin_str] = inner_dict
    lin_str = "{0}_intercept".format(species)
    value = info_df.iloc[test_row, :][lin_str]
    inner_dict = {'custom_object_name': 'lin_param_df',
                  'function': mod_lin_param_df,
                  'kwargs': {'mini_species': species,
                             'mini_lin_param': 'intercept'},
                  'input_value': value
                  }
    info_dict[lin_str] = inner_dict

info_dict.update(pitzer_params_dict)
estimator = llepe.LLEPE(**estimator_params)
estimator.set_custom_objects_dict({'lin_param_df': lin_param_df})
estimator.update_custom_objects_dict(info_dict)
estimator.update_xml(info_dict,
                     dependant_params_dict=dependant_params_dict)
compared_value = 'La_org_eq'
plot_title = None
legend = True
predicted_dict = estimator.get_predicted_dict()
exp_df = estimator.get_exp_df()
pred = pd.DataFrame(predicted_dict)[compared_value].fillna(0).values
meas = exp_df[compared_value].fillna(0).values
name_breakdown = re.findall('[^_\W]+', compared_value)
compared_species = name_breakdown[0]
data_labels = list(labeled_data['label'])
if compared_species == 'h':
    feed_molarity = exp_df['h_i'].fillna(0).values
elif compared_species == 'z':
    feed_molarity = exp_df['z_i'].fillna(0).values
else:
    feed_molarity = exp_df[
        '{0}_aq_i'.format(compared_species)].fillna(0).values
combined_df = pd.DataFrame({'pred': pred,
                            'meas': meas,
                            'label': data_labels,
                            'feed_molarity': feed_molarity})
combined_df = combined_df[(combined_df['feed_molarity'] != 0)]
meas = combined_df['meas'].values
pred = combined_df['pred'].values

min_data = np.min([pred, meas])
max_data = np.max([pred, meas])
min_max_data = np.array([min_data, max_data])

if compared_species == 'h':
    default_title = '$H^+$ eq. conc. (mol/L)'
elif compared_species == 'z':
    default_title = '{0} eq. conc. (mol/L)'.format(extractant_name)
else:
    phase = name_breakdown[1]
    if phase == 'aq':
        extracted_species_charge = extracted_species_charges[
            extracted_species_list.index(
                compared_species)]
        default_title = '$%s^{%d+}$ eq. conc. (mol/L)' \
                        % (compared_species, extracted_species_charge)
    elif phase == 'd':
        default_title = '{0} distribution ratio'.format(
            compared_species)
    else:
        default_title = '{0} complex eq. conc. (mol/L)'.format(
            compared_species)
fig, ax = plt.subplots(figsize=(8, 6))

if isinstance(data_labels, list):
    # unique_labels = list(set(data_labels))
    unique_labels = ['Li (1987)',
                     'Kim (2012)',
                     'Formiga (2016)',
                     'Banda (2014)',
                     ]
    color_list = ['r', 'g', 'b', 'm']
    marker_list = ['o', 's', 'P', 'X', ]
    for ind, label in enumerate(unique_labels):
        filtered_data = combined_df[combined_df['label'] == label]
        filtered_meas = filtered_data['meas']
        filtered_pred = filtered_data['pred']
        if len(filtered_pred) != 0:
            ax.scatter(filtered_meas,
                       filtered_pred,
                       label=label,
                       color=color_list[ind],
                       marker=marker_list[ind])
    if legend:
        ax.legend(loc=4)
ax.plot(min_max_data, min_max_data, color="b", label="")

ax.text(min_max_data[0],
        min_max_data[1] * 0.9,
        '$R^2$={0:.2f}'.format(estimator.r_squared(compared_value)))

ax.set(xlabel='Measured', ylabel='Predicted')
if plot_title is None:
    ax.set_title(default_title)
set_size(8, 6)
plt.tight_layout()
plt.show()
# exp_data = estimator.get_exp_df()
# feed_cols = []
# for col in exp_data.columns:
#     if 'aq_i' in col:
#         feed_cols.append(col)
# exp_data['total_re'] = exp_data[feed_cols].sum(axis=1)
# label_list = []
# for index, row in exp_data[feed_cols].iterrows():
#     bool_list = list((row > 0).values)
#     label = ''
#     for species, el in zip(species_list, bool_list):
#         if el:
#             label = '{0}-{1}'.format(label, species)
#     label = label[1:]
#     label_list.append(label)
# r2s = ""
# for species in species_list:
#     # if species=='La':
#     # save_name = 'outputs' \
#     #             '/parity_iterative_fitter_{0}_org_eq'.format(species)
#     save_name = None
#     fig, ax = estimator.parity_plot('{0}_org_eq'.format(species),
#                                     c_data=
#                                     exp_data['total_re'].values,
#                                     c_label='Feed total RE '
#                                             'molarity (mol/L)',
#                                     print_r_squared=False,
#                                     plot_title='')
#     ax.plot([0, 0.05], [0, 0.05], c='b')
#     ax.text(0.01, 0.04,
#             '$R^2$={0:.2f}'.format(estimator.r_squared(
#                 '{0}_org_eq'.format(species))))
#     ax.set_xlim((0, 0.05))
#     ax.set_ylim((0, 0.05))
#     r2s += str(estimator.r_squared('{0}_org_eq'.format(species))) + ','
#
#     # fig, ax = estimator.parity_plot('{0}_org_eq'.format(species),
#     #                                 data_labels=list(labeled_data['label']),
#     #                                 print_r_squared=True,
#     #                                 save_path=save_name)
#     # ax.legend(loc=4)
# pred_df = pd.DataFrame(estimator.get_predicted_dict())
# new_cols = []
# for col in pred_df.columns:
#     new_cols.append("pred_{0}".format(col))
# pred_df.columns = new_cols
# new_cols = ['label',
#             'h_i',
#             'h_eq',
#             'z_i',
#             'z_eq'
#             ]
# for species in species_list:
#     new_cols.append("{0}_aq_i".format(species))
#     new_cols.append("{0}_aq_eq".format(species))
#     new_cols.append("{0}_d_eq".format(species))
# labeled_data.columns = new_cols
# total_df = labeled_data.join(pred_df)
# total_df.to_csv('if_mse_total_df.csv')
# short_info_dict = {}
# for key, value in info_dict.items():
#     short_info_dict[key] = value['input_value']
# with open("outputs/iterative_fitter_short_info_dict.txt", 'w') as file:
#     json.dump(short_info_dict, file)
