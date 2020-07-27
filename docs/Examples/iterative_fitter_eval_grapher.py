import llepe
import pandas as pd
import numpy as np
import json
import matplotlib as plt
import matplotlib

font = {'family': 'sans serif',
        'size': 24}
matplotlib.rc('font', **font)
plt.rc('xtick', labelsize=18)
plt.rc('ytick', labelsize=18)
plt.rcParams['lines.linewidth'] = 4
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


info_df = pd.read_csv('outputs/iterative_fitter_w_mse_output.csv')
test_row = 3
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
exp_data = estimator.get_exp_df()
feed_cols = []
for col in exp_data.columns:
    if 'aq_i' in col:
        feed_cols.append(col)
exp_data['total_re'] = exp_data[feed_cols].sum(axis=1)
label_list = []
for index, row in exp_data[feed_cols].iterrows():
    bool_list = list((row > 0).values)
    label = ''
    for species, el in zip(species_list, bool_list):
        if el:
            label = '{0}-{1}'.format(label, species)
    label = label[1:]
    label_list.append(label)
r2s = ""
for species in species_list:
    # if species=='La':
    # save_name = 'outputs' \
    #             '/parity_iterative_fitter_{0}_org_eq'.format(species)
    save_name = None
    # fig, ax = estimator.parity_plot('{0}_org_eq'.format(species),
    #                                 c_data='z_i',
    #                                 c_label='Feed total RE '
    #                                         'molarity (mol/L)',
    #                                 print_r_squared=True,
    #                                 save_path=save_name)
    r2s += str(estimator.r_squared('{0}_org_eq'.format(species))) + ','

    fig, ax = estimator.parity_plot('{0}_org_eq'.format(species),
                                    data_labels=label_list,
                                    print_r_squared=True,
                                    save_path=save_name)
    ax.legend(loc=4)
pred_df = pd.DataFrame(estimator.get_predicted_dict())
new_cols = []
for col in pred_df.columns:
    new_cols.append("pred_{0}".format(col))
pred_df.columns = new_cols
new_cols = ['label',
            'h_i',
            'h_eq',
            'z_i',
            'z_eq'
            ]
for species in species_list:
    new_cols.append("{0}_aq_i".format(species))
    new_cols.append("{0}_aq_eq".format(species))
    new_cols.append("{0}_d_eq".format(species))
labeled_data.columns = new_cols
total_df = labeled_data.join(pred_df)
# total_df.to_csv('if_mse_total_df.csv')
# short_info_dict = {}
# for key, value in info_dict.items():
#     short_info_dict[key] = value['input_value']
# with open("outputs/iterative_fitter_short_info_dict.txt", 'w') as file:
#     json.dump(short_info_dict, file)
