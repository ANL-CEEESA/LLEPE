import json
from unittest import TestCase

import numpy as np
import pyswarms as ps
from reeps import REEPS1

with open('one_ree_settings.txt') as file:
    testing_params = json.load(file)

beaker = REEPS1(**testing_params)


# def new_obj(predicted_dict, meas_df, epsilon):
#     meas_cols = list(meas_df)
#     pred_keys = list(predicted_dict.keys())
#     meas = meas_df[meas_cols[2]]
#     pred = (predicted_dict['re_org'] + epsilon) / (predicted_dict['re_aq'] + epsilon)
#     log_pred = np.log10(pred)
#     log_meas = np.log10(meas)
#     obj = np.sum((log_pred - log_meas) ** 2)
#     return obj
# #
# #
# # def new_obj(ping):
# #     print(ping)
# beaker.set_objective_function(new_obj)
# objective_kwargs = {"epsilon": 1e-14}
# beaker.set
# noinspection PyUnusedLocal
def optimizer(func, x_guess):
    lb = np.array([1e-1])
    ub = np.array([1e1])
    bounds = (lb, ub)
    options = {'c1': 1e-3, 'c2': 1e-3, 'w': 0.9}
    mini_optimizer = ps.single.global_best.GlobalBestPSO(n_particles=100, dimensions=1,
                                                         options=options, bounds=bounds)
    f_opt, x_opt = mini_optimizer.optimize(func, iters=100)

    return x_opt


minimizer_kwargs = {"method": 'SLSQP',
                    "bounds": [(1e-1, 1e1)],
                    "constraints": (),
                    "options": {'disp': True, 'maxiter': 1000, 'ftol': 1e-6}}
# est_enthalpy = beaker.fit(optimizer=optimizer)
est_enthalpy = beaker.fit()
print(est_enthalpy)


beaker.update_xml(est_enthalpy)
# beaker.parity_plot()
print(beaker.r_squared())
# class TestREEPS1(TestCase):
#     def test_slsqp_optimizer(self):
#         self.fail()
#
#     def test_log_mean_squared_error(self):
#         self.fail()
#
#     def test_get_exp_df(self):
#         self.fail()
#
#     def test_set_exp_df(self):
#         self.fail()
#
#     def test_get_phases(self):
#         self.fail()
#
#     def test_set_phases(self):
#         self.fail()
#
#     def test_get_opt_dict(self):
#         self.fail()
#
#     def test_set_opt_dict(self):
#         self.fail()
#
#     def test_get_aq_solvent_name(self):
#         self.fail()
#
#     def test_set_aq_solvent_name(self):
#         self.fail()
#
#     def test_get_extractant_name(self):
#         self.fail()
#
#     def test_set_extractant_name(self):
#         self.fail()
#
#     def test_get_diluant_name(self):
#         self.fail()
#
#     def test_set_diluant_name(self):
#         self.fail()
#
#     def test_get_complex_names(self):
#         self.fail()
#
#     def test_set_complex_names(self):
#         self.fail()
#
#     def test_get_rare_earth_ion_names(self):
#         self.fail()
#
#     def test_set_rare_earth_ion_names(self):
#         self.fail()
#
#     def test_get_aq_solvent_rho(self):
#         self.fail()
#
#     def test_set_aq_solvent_rho(self):
#         self.fail()
#
#     def test_get_extractant_rho(self):
#         self.fail()
#
#     def test_set_extractant_rho(self):
#         self.fail()
#
#     def test_get_diluant_rho(self):
#         self.fail()
#
#     def test_set_diluant_rho(self):
#         self.fail()
#
#     def test_set_in_moles(self):
#         self.fail()
#
#     def test_get_in_moles(self):
#         self.fail()
#
#     def test_set_objective_function(self):
#         self.fail()
#
#     def test_get_objective_function(self):
#         self.fail()
#
#     def test_set_optimizer(self):
#         self.fail()
#
#     def test_get_optimizer(self):
#         self.fail()
#
#     def test_update_predicted_dict(self):
#         self.fail()
#
#     def test_get_predicted_dict(self):
#         self.fail()
#
#     def test__internal_objective(self):
#         self.fail()
#
#     def test_fit(self):
#         self.fail()
#
#     def test_update_xml(self):
#         self.fail()
#
#     def test_parity_plot(self):
#         self.fail()
#
#     def test_r_squared(self):
#         self.fail()
