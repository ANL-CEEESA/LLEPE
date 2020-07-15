from scipy.optimize import curve_fit
import llepe
import pandas as pd
import numpy as np


def linear(x, a, b):
    return a * x + b


species_list = 'Nd,Pr,Ce,La,Dy,Sm,Y'.split(',')
pitzer_param_list = ['beta0', 'beta1', 'Cphi']
meas_pitzer_param_df = pd.read_csv("../../data/csvs/may_pitzer_params.csv")
labeled_data = pd.read_csv("../../data/csvs/"
                           "zeroes_removed_PC88A_HCL_NdPrCeLaDySmY.csv")
exp_data = labeled_data.drop(labeled_data.columns[0], axis=1)
xml_file = "PC88A_HCL_NdPrCeLaDySmY_w_pitzer.xml"
eps = 1e-4
mini_eps = 1e-8
x_guesses = [[-5178500.0, -1459500.0],
             [-5178342.857142857, -1460300.0],
             [-5178342.857142857, -1459500.0],
             [-5178342.857142857, -1458300.0],
             [-5178185.714285715, -1459900.0],
             [-5178185.714285715, -1459500.0],
             [-5178185.714285715, -1459100.0],
             [-5178185.714285715, -1458300.0],
             [-5178028.571428572, -1459900.0],
             [-5178028.571428572, -1459100.0],
             [-5178028.571428572, -1458300.0],
             [-5177557.142857143, -1459900.0],
             [-5177400.0, -1460300.0]]
pitzer_guess_df = meas_pitzer_param_df.copy()
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
    best_obj = 1e20
    output_dict['iter'].append(i)
    for species in species_list:
        lower_species = species.lower()
        opt_values = {
            '(HA)2(org)_h0': [],
            '{0}(H(A)2)3(org)_h0'.format(species): [],
            'beta0': [],
            'beta1': [],
            'Cphi': [],
            'obj_value': [],
            'guess': []}
        for x_guess in x_guesses:
            info_dict = {'(HA)2(org)_h0': {'upper_element_name': 'species',
                                           'upper_attrib_name': 'name',
                                           'upper_attrib_value': '(HA)2(org)',
                                           'lower_element_name': 'h0',
                                           'lower_attrib_name': None,
                                           'lower_attrib_value': None,
                                           'input_format': '{0}',
                                           'input_value': x_guess[1]},
                         '{0}(H(A)2)3(org)_h0'.format(species): {
                             'upper_element_name': 'species',
                             'upper_attrib_name': 'name',
                             'upper_attrib_value': '{0}(H(A)2)3(org)'.format(
                                 species),
                             'lower_element_name': 'h0',
                             'lower_attrib_name': None,
                             'lower_attrib_value': None,
                             'input_format': '{0}',
                             'input_value': x_guess[0]},
                         }
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
            llepe_params = {
                'exp_data': exp_data,
                'phases_xml_filename': xml_file,
                'opt_dict': info_dict,
                'phase_names': ['HCl_electrolyte', 'PC88A_liquid'],
                'aq_solvent_name': 'H2O(L)',
                'extractant_name': '(HA)2(org)',
                'diluant_name': 'dodecane',
                'complex_names': ['{0}(H(A)2)3(org)'.format(species)
                                  for species in species_list],
                'extracted_species_ion_names': ['{0}+++'.format(species)
                                                for species in species_list],
                'aq_solvent_rho': 1000.0,
                'extractant_rho': 960.0,
                'diluant_rho': 750.0,
                'objective_function': llepe.lmse_perturbed_obj,
                'optimizer': optimizer,
                'temp_xml_file_path': 'outputs/temp.xml'
            }
            estimator = llepe.LLEPE(**llepe_params)
            estimator.update_xml(llepe_params['opt_dict'])
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
            info1 = [opt_dict[key]['input_value'] for key in keys]
            info1.append(obj_value)
            info1.append(x_guess)
            opt_values_keys = opt_values.keys()
            for ind, key in enumerate(opt_values_keys):
                opt_values[key].append(info1[ind])
        opt_value_df = pd.DataFrame(opt_values)
        p_opt, p_cov = curve_fit(linear,
                                 opt_value_df['(HA)2(org)_h0'].values,
                                 opt_value_df['{0}(H(A)2)3(org)_h0'.format(
                                     species)].values)
        slope, intercept = p_opt
        output_dict['{0}_slope'.format(species)].append(slope)
        output_dict['{0}_intercept'.format(species)].append(intercept)
        min_h0_df = opt_value_df[
            opt_value_df['(HA)2(org)_h0']
            == opt_value_df['(HA)2(org)_h0'].min()]
        update_pitzer_dict = {}
        for pitzer_param in pitzer_param_list:
            key_name = '{0}_{1}'.format(species, pitzer_param)
            output_dict[key_name].append(min_h0_df[pitzer_param].values[0])

            inner_dict = {'upper_element_name': 'binarySaltParameters',
                          'upper_attrib_name': 'cation',
                          'upper_attrib_value':
                              '{0}+++'.format(species),
                          'lower_element_name': pitzer_param,
                          'lower_attrib_name': None,
                          'lower_attrib_value': None,
                          'input_format': ' {0}, 0.0, 0.0, 0.0, 0.0 ',
                          'input_value':
                              min_h0_df[pitzer_param].values[0]
                          }
            update_pitzer_dict['{0}_{1}'.format(
                species, pitzer_param)] = inner_dict
        estimator.update_xml(update_pitzer_dict)

    pitzer_guess_dict = {'species': [],
                         'beta0': [],
                         'beta1': [],
                         'Cphi': []}
    for species in species_list:
        pitzer_guess_dict['species'].append(species)
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
    pitzer_guess_df = pd.DataFrame(pitzer_guess_dict)

    output_dict['best_obj'].append(best_obj)
    output_df = pd.DataFrame(output_dict)
    old_row = output_df.iloc[-2, :].values[3:]
    new_row = output_df.iloc[-1, :].values[3:]
    rel_diff = np.sum(np.abs(new_row - old_row) / np.abs(old_row))
    output_dict['rel_diff'].append(rel_diff)
    output_df = pd.DataFrame(output_dict)
    output_df.to_csv('outputs/iterative_fitter_output_df.csv')
