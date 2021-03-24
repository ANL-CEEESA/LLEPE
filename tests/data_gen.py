#  LLEPE: Liquid-Liquid Equilibrium Parameter Estimator
#  Copyright (C) 2020, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See LICENSE for more details.

from llepe import LLEPE
import json

opt_dict = {'(HA)2(org)_h0': {'upper_element_name': 'species',
                              'upper_attrib_name': 'name',
                              'upper_attrib_value': 'Nd(H(A)2)3(org)',
                              'lower_element_name': 'h0',
                              'lower_attrib_name': None,
                              'lower_attrib_value': None,
                              'input_format': '{0}',
                              'input_value': -4.7e6}}

searcher_parameters = {'exp_data': 'Nd_exp_data.csv',
                       'phases_xml_filename':
                           'twophase.xml',
                       'opt_dict': opt_dict,
                       'phase_names': ['HCl_electrolyte',
                                       'PC88A_liquid'],
                       'aq_solvent_name': 'H2O(L)',
                       'extractant_name': '(HA)2(org)',
                       'diluant_name': 'dodecane',
                       'complex_names':
                           ['Nd(H(A)2)3(org)'],
                       'extracted_species_ion_names': ['Nd+++'],
                       'aq_solvent_rho': 1000.0,
                       'extractant_rho': 960.0,
                       'diluant_rho': 750.0,
                       'temp_xml_file_path': 'temp1.xml'}
searcher = LLEPE(**searcher_parameters)
searcher.update_xml(searcher_parameters['opt_dict'])
predicted_dict1 = searcher.get_predicted_dict()


def array_to_list_in_dict(dictionary):
    new_dictionary = {}
    for key, value in dictionary.items():
        new_dictionary[key] = list(value)
    return new_dictionary


predicted_dict1 = array_to_list_in_dict(predicted_dict1)
in_moles = searcher.get_in_moles().to_dict('list')

est_enthalpy, obj_value = searcher.fit()
searcher.update_xml(est_enthalpy)
predicted_dict2 = searcher.get_predicted_dict()
predicted_dict2 = array_to_list_in_dict(predicted_dict2)
r2 = searcher.r_squared()
validation_values = {'predicted_dict1': predicted_dict1,
                     'in_moles': in_moles,
                     'est_enthalpy': est_enthalpy,
                     'predicted_dict2': predicted_dict2,
                     'r2': r2}
with open("validation_values.txt", "w") as write_file:
    json.dump(validation_values, write_file)

with open("validation_parameters.txt", 'w') as write_file:
    json.dump(searcher_parameters, write_file)
print(r2)
