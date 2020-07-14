import json
from llepe import LLEPE
import pkg_resources

validation_parameters_filename = pkg_resources.resource_filename(
    'llepe',
    r'..\tests\validation_parameters.txt')
validation_values_filename = pkg_resources.resource_filename(
    'llepe',
    r'..\tests\validation_values.txt')

with open(validation_parameters_filename) as file:
    validation_params = json.load(file)
with open(validation_values_filename) as file:
    validation_values = json.load(file)
searcher = LLEPE(**validation_params)


def array_to_list_in_dict(dictionary):
    new_dictionary = {}
    for key, value in dictionary.items():
        new_dictionary[key] = list(value)
    return new_dictionary


def test_init():
    searcher.update_xml(validation_params['opt_dict'])
    predicted_dict1 = searcher.get_predicted_dict()
    predicted_dict1 = array_to_list_in_dict(predicted_dict1)
    in_moles = searcher.get_in_moles().to_dict('list')
    assert predicted_dict1 == validation_values['predicted_dict1'], \
        "Prediction dicts are not equal. Error with set_in_moles, "
    "update_predicted_dict, or changed xmls, or changed data"
    assert in_moles == validation_values['in_moles'], \
        "In_moles are different. Error with set_in_moles or data"
    return None


def test_fit():
    est_enthalpy, obj_value = searcher.fit()
    searcher.update_xml(est_enthalpy)
    predicted_dict2 = searcher.get_predicted_dict()
    predicted_dict2 = array_to_list_in_dict(predicted_dict2)
    r2 = searcher.r_squared()
    assert est_enthalpy == validation_values['est_enthalpy'], \
        "estimated enthalpy is not equal. Check fit method"
    assert predicted_dict2 == validation_values['predicted_dict2'], \
        "Prediction dicts are not equal. Error with set_in_moles, "
    "update_predicted_dict, or changed xmls, or changed data"
    assert r2 == validation_values['r2'], "r-squared value is off, check " \
                                          "r_squared method"
