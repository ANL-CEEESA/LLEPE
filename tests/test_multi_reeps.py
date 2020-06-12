import json
import numpy as np
import sys
sys.path.append('../')
from reeps import REEPS

with open('multi_ree_settings.txt') as file:
    testing_params = json.load(file)

beaker = REEPS(**testing_params)


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
# beaker.parity_plot('Nd_d_eq', print_r_squared=True)
print(beaker.r_squared())
