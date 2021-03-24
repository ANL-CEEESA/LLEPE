#  LLEPE: Liquid-Liquid Equilibrium Parameter Estimator
#  Copyright (C) 2020, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See LICENSE for more details.

# This file tests adding 0.1 M NaCl to the feed.
import llepe
import pandas as pd
import numpy as np
import json
import matplotlib as plt
import matplotlib
import cantera as ct


class ModLLEPE(llepe.LLEPE):
    def __init__(self,
                 exp_data,
                 phases_xml_filename,
                 phase_names,
                 aq_solvent_name,
                 extractant_name,
                 diluant_name,
                 complex_names,
                 extracted_species_ion_names,
                 extracted_species_list=None,
                 aq_solvent_rho=None,
                 extractant_rho=None,
                 diluant_rho=None,
                 opt_dict=None,
                 objective_function='Log-MSE',
                 optimizer='scipy_minimize',
                 temp_xml_file_path=None,
                 dependant_params_dict=None,
                 custom_objects_dict=None,
                 nacl_molarity=0):
        self.nacl_molarity = nacl_molarity
        super().__init__(exp_data,
                         phases_xml_filename,
                         phase_names,
                         aq_solvent_name,
                         extractant_name,
                         diluant_name,
                         complex_names,
                         extracted_species_ion_names,
                         extracted_species_list,
                         aq_solvent_rho,
                         extractant_rho,
                         diluant_rho,
                         opt_dict,
                         objective_function,
                         optimizer,
                         temp_xml_file_path,
                         dependant_params_dict,
                         custom_objects_dict)


    def set_in_moles(self, feed_vol):
        """Function that initializes mole fractions to input feed_vol

                This function is called at initialization

                Sets in_moles to a pd.DataFrame containing initial mole fractions

                Columns for species and rows for different experiments

                This function also calls update_predicted_dict

                :param feed_vol: (float) feed volume of mixture (L)
                """
        phases_copy = self._phases.copy()
        exp_df = self._exp_df.copy()
        solvent_name = self._aq_solvent_name
        extractant_name = self._extractant_name
        diluant_name = self._diluant_name
        solvent_rho = self._aq_solvent_rho
        extractant_rho = self._extractant_rho
        diluant_rho = self._diluant_rho
        extracted_species_names = self._extracted_species_ion_names
        extracted_species_list = self._extracted_species_list

        mixed = ct.Mixture(phases_copy)
        aq_ind = None
        solvent_ind = None
        for ind, phase in enumerate(phases_copy):
            if solvent_name in phase.species_names:
                aq_ind = ind
                solvent_ind = phase.species_names.index(solvent_name)
        if aq_ind is None:
            raise Exception('Solvent "{0}" not found \
                                        in xml file'.format(solvent_name))

        if aq_ind == 0:
            org_ind = 1
        else:
            org_ind = 0
        self._aq_ind = aq_ind
        self._org_ind = org_ind
        extractant_ind = phases_copy[org_ind].species_names.index(
            extractant_name)
        diluant_ind = phases_copy[org_ind].species_names.index(
            diluant_name)

        extracted_species_ind_list = [
            phases_copy[aq_ind].species_names.index(
                extracted_species_name)
            for extracted_species_name in extracted_species_names]
        extracted_species_charges = np.array(
            [phases_copy[aq_ind].species(
                extracted_species_ind).charge
             for extracted_species_ind in extracted_species_ind_list])
        self._extracted_species_charges = extracted_species_charges

        mix_aq = mixed.phase(aq_ind)
        mix_org = mixed.phase(org_ind)
        solvent_mw = mix_aq.molecular_weights[solvent_ind]  # g/mol
        extractant_mw = mix_org.molecular_weights[extractant_ind]
        diluant_mw = mix_org.molecular_weights[diluant_ind]
        if solvent_rho is None:
            solvent_rho = mix_aq(aq_ind).partial_molar_volumes[
                              solvent_ind] / solvent_mw * 1e6  # g/L
            self._aq_solvent_rho = solvent_rho
        if extractant_rho is None:
            extractant_rho = mix_org(org_ind).partial_molar_volumes[
                                 extractant_ind] / extractant_mw * 1e6
            self._extractant_rho = extractant_rho
        if diluant_rho is None:
            diluant_rho = mix_org(org_ind).partial_molar_volumes[
                              extractant_ind] / extractant_mw * 1e6
            self._diluant_rho = diluant_rho

        in_moles_data = []
        aq_phase_solvent_moles = feed_vol * solvent_rho / solvent_mw
        for index, row in exp_df.iterrows():
            h_plus_moles = feed_vol * row['h_i']
            hydroxide_ions = 0
            extracted_species_moles = np.array(
                [feed_vol * row['{0}_aq_i'.format(
                    extracted_species)]
                 for extracted_species in extracted_species_list])
            extracted_species_charge_sum = np.sum(
                extracted_species_charges * extracted_species_moles)
            chlorine_moles = extracted_species_charge_sum + h_plus_moles
            extractant_moles = feed_vol * row['z_i']
            extractant_vol = extractant_moles * extractant_mw / extractant_rho
            diluant_vol = feed_vol - extractant_vol
            diluant_moles = diluant_vol * diluant_rho / diluant_mw
            complex_moles = np.zeros(len(extracted_species_list))

            species_moles_aq = [aq_phase_solvent_moles,
                                h_plus_moles,
                                hydroxide_ions,
                                chlorine_moles]
            species_moles_aq.extend(list(extracted_species_moles))
            species_moles_aq.append(self.nacl_molarity * feed_vol)
            species_moles_aq[3] += self.nacl_molarity * feed_vol
            species_moles_org = [extractant_moles, diluant_moles]
            species_moles_org.extend(list(complex_moles))
            if aq_ind == 0:
                species_moles = species_moles_aq + species_moles_org
            else:
                species_moles = species_moles_org + species_moles_aq
            in_moles_data.append(species_moles)

        self._in_moles = pd.DataFrame(
            in_moles_data, columns=mixed.species_names)
        self.update_predicted_dict()
        return None


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


info_df = pd.read_csv('outputs/multi_only_iterative_fitter_output.csv')
test_row = -1
pitzer_params_filename = "../../data/jsons/min_h0_pitzer_params.txt"
with open(pitzer_params_filename) as file:
    pitzer_params_dict = json.load(file)
pitzer_params_df = pd.DataFrame(pitzer_params_dict)
species_list = 'Nd,Pr,Ce,La,Dy,Sm,Y'.split(',')
pitzer_param_list = ['beta0', 'beta1']
labeled_data = pd.read_csv("../../data/csvs/"
                           "no_formiga_or_5_oa_PC88A_HCL_NdPrCeLaDySmY.csv")
labeled_data = labeled_data.sort_values(['Feed Pr[M]', 'Feed Ce[M]'],
                                        ascending=True)
exp_data = labeled_data.drop(labeled_data.columns[0], axis=1)
xml_file = "test_PC88A_HCL_NdPrCeLaDySmY_w_pitzer.xml"
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
                    'objective_function': llepe.lmse_perturbed_obj,
                    'nacl_molarity': 0
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
estimator = ModLLEPE(**estimator_params)
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
                                    data_labels=list(labeled_data['label']),
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
