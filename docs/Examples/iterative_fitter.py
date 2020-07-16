import llepe
import pandas as pd
import numpy as np
import json


def mod_lin_param_df(lp_df, input_val, mini_species, mini_lin_param):
    new_lp_df = lp_df.copy()
    index = new_lp_df.index[new_lp_df['species'] == mini_species].tolist()[0]
    new_lp_df.at[index, mini_lin_param] = input_val
    return new_lp_df


species_list = 'Nd,Pr,Ce,La,Dy,Sm,Y'.split(',')
pitzer_param_list = ['beta0', 'beta1']
lin_param_list = ['slope', 'intercept']
meas_pitzer_param_df = pd.read_csv("../../data/csvs/may_pitzer_params.csv")
pitzer_params_filename = "../../data/jsons/min_h0_pitzer_params.txt"
with open(pitzer_params_filename) as file:
    pitzer_params_dict = json.load(file)
ext_h0_filename = "../../data/jsons/min_h0_guess_ext_h0.txt"
with open(ext_h0_filename) as file:
    ext_h0_dict = json.load(file)
labeled_data = pd.read_csv("../../data/csvs/"
                           "zeroes_removed_PC88A_HCL_NdPrCeLaDySmY.csv")
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
estimator = llepe.LLEPE(**estimator_params)


def ext_to_complex(h0, custom_obj_dict, mini_species):
    linear_params = custom_obj_dict['lin_param_df']
    row = linear_params[linear_params['species'] == mini_species]
    return row['slope'].values[0] * h0[0] + row['intercept'].values[0]


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
estimator.update_xml(pitzer_params_dict)
estimator.set_custom_objects_dict({'lin_param_df': lin_param_df})
estimator.set_dependant_params_dict(dependant_params_dict)
estimator.update_xml(ext_h0_dict,
                     dependant_params_dict=dependant_params_dict)
eps = 1e-4
mini_eps = 1e-8
pitzer_guess_dict = {'species': [],
                     'beta0': [],
                     'beta1': []}
for species in species_list:
    pitzer_guess_dict['species'].append(species)
    for param in pitzer_param_list:
        mini_dict = pitzer_params_dict['{0}_{1}'.format(species, param)]
        value = mini_dict['input_value']
        pitzer_guess_dict[param].append(value)
pitzer_guess_df = pd.DataFrame(pitzer_guess_dict)
ext_h0_guess = ext_h0_dict['(HA)2(org)_h0']['input_value']
lin_guess_df = lin_param_df.copy()

ignore_list = []
optimizer = 'scipy_minimize'
output_dict = {'iter': [0],
               'best_obj': [1e20],
               'rel_diff': [1e20]}
for species in species_list:
    output_dict['{0}_slope'.format(species)] = [1e20]
    output_dict['{0}_intercept'.format(species)] = [1e20]
    for pitzer_param in pitzer_param_list:
        output_dict['{0}_{1}'.format(species, pitzer_param)] = [1e20]
i = 0
rel_diff = 1000
while rel_diff > 1e-4:
    i += 1
    print(i)
    best_obj = 1e20
    output_dict['iter'].append(i)
    for species in species_list:
        print(species)
        lower_species = species.lower()
        info_dict = {'(HA)2(org)_h0': {'upper_element_name': 'species',
                                       'upper_attrib_name': 'name',
                                       'upper_attrib_value': '(HA)2(org)',
                                       'lower_element_name': 'h0',
                                       'lower_attrib_name': None,
                                       'lower_attrib_value': None,
                                       'input_format': '{0}',
                                       'input_value': ext_h0_guess}}

        for pitzer_param in pitzer_param_list:
            if '{0}_{1}'.format(species, pitzer_param) not in ignore_list:
                pitzer_row = pitzer_guess_df[
                    pitzer_guess_df['species'] == species]
                inner_dict = {'upper_element_name': 'binarySaltParameters',
                              'upper_attrib_name': 'cation',
                              'upper_attrib_value':
                                  '{0}+++'.format(species),
                              'lower_element_name': pitzer_param,
                              'lower_attrib_name': None,
                              'lower_attrib_value': None,
                              'input_format': ' {0}, 0.0, 0.0, 0.0, 0.0 ',
                              'input_value':
                                  pitzer_row[pitzer_param].values[0]
                              }
                info_dict['{0}_{1}'.format(
                    species, pitzer_param)] = inner_dict
        for lin_param in lin_param_list:
            if '{0}_{1}'.format(species, lin_param) not in ignore_list:
                lin_row = lin_guess_df[lin_guess_df['species'] == species]
                inner_dict = {'custom_object_name': 'lin_param_df',
                              'function': mod_lin_param_df,
                              'kwargs': {'mini_species': species,
                                         'mini_lin_param': lin_param},
                              'input_value': lin_row[lin_param].values[0]
                              }
                info_dict['{0}_{1}'.format(
                    species, lin_param)] = inner_dict
        estimator.set_opt_dict(info_dict)
        estimator.update_custom_objects_dict(info_dict)
        estimator.update_xml(info_dict)
        obj_kwargs = {'species_list': [species], 'epsilon': 1e-100}
        bounds = [(1e-1, 1e1)] * len(info_dict)
        optimizer_kwargs = {"method": 'l-bfgs-b',
                            "bounds": bounds}
        opt_dict, obj_value = estimator.fit(
            objective_kwargs=obj_kwargs,
            optimizer_kwargs=optimizer_kwargs)
        if obj_value < best_obj:
            best_obj = obj_value
        keys = list(opt_dict.keys())
        for lin_param in lin_param_list:
            mini_dict = opt_dict['{0}_{1}'.format(species, lin_param)] 
            value = mini_dict['input_value']
            output_dict['{0}_{1}'.format(species, lin_param)].append(value)
        for pitzer_param in pitzer_param_list:
            mini_dict = opt_dict['{0}_{1}'.format(species, pitzer_param)]
            value = mini_dict['input_value']
            output_dict['{0}_{1}'.format(species, pitzer_param)].append(value)
        estimator.update_custom_objects_dict(info_dict)
        estimator.update_xml(opt_dict)
    pitzer_guess_dict = {'species': [],
                         'beta0': [],
                         'beta1': [],
                         }
    lin_guess_dict = {'species': [],
                      'slope': [],
                      'intercept': []}
    for species in species_list:
        pitzer_guess_dict['species'].append(species)
        lin_guess_dict['species'].append(species)
        for pitzer_param in pitzer_param_list:
            value_list = output_dict['{0}_{1}'.format(species, pitzer_param)]
            value = value_list[-1]
            pitzer_guess_dict[pitzer_param].append(value)
            if i > 2:
                mini_rel_diff1 = np.abs(value_list[-1]
                                        - value_list[-2]) / (
                                     np.abs(value_list[-2]))
                mini_rel_diff2 = np.abs(value_list[-2] - value_list[-3]) / (
                    np.abs(value_list[-3]))
                if mini_rel_diff1 < mini_eps and mini_rel_diff2 < mini_eps:
                    ignore_list.append('{0}_{1}'.format(species, pitzer_param))
        for lin_param in lin_param_list:
            value_list = output_dict['{0}_{1}'.format(species, lin_param)]
            value = value_list[-1]
            lin_guess_dict[lin_param].append(value)
            if i > 2:
                mini_rel_diff1 = np.abs(value_list[-1]
                                        - value_list[-2]) / (
                                     np.abs(value_list[-2]))
                mini_rel_diff2 = np.abs(value_list[-2] - value_list[-3]) / (
                    np.abs(value_list[-3]))
                if mini_rel_diff1 < mini_eps and mini_rel_diff2 < mini_eps:
                    ignore_list.append('{0}_{1}'.format(species, lin_param))
    pitzer_guess_df = pd.DataFrame(pitzer_guess_dict)
    lin_guess_df = pd.DataFrame(lin_guess_dict)

    output_dict['best_obj'].append(best_obj)
    output_dict['rel_diff'].append(100)
    output_df = pd.DataFrame(output_dict)
    old_row = output_df.iloc[-2, :].values[3:]
    new_row = output_df.iloc[-1, :].values[3:]
    rel_diff = np.sum(np.abs(new_row - old_row) / np.abs(old_row))
    del(output_dict['rel_diff'][-1])
    output_dict['rel_diff'].append(rel_diff)
    output_df = pd.DataFrame(output_dict)
    output_df.to_csv('outputs/iterative_fitter_output.csv')
