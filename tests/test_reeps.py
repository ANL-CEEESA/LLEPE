import json
import sys

sys.path.append('../')
from reeps import REEPS


with open('one_comp_settings.txt') as file:
    testing_params = json.load(file)
beaker = REEPS(**testing_params)

minimizer_kwargs = {"method": 'SLSQP',
                    "bounds": [(1e-1, 1e1)],
                    "constraints": (),
                    "options": {'disp': True, 'maxiter': 1000, 'ftol': 1e-6}}
est_enthalpy = beaker.fit(minimizer_kwargs)
print(est_enthalpy)
beaker.update_xml(est_enthalpy)
beaker.parity_plot()
