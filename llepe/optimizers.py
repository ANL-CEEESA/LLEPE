#  LLEPE: Liquid-Liquid Equilibrium Parameter Estimator
#  Copyright (C) 2020, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See LICENSE for more details.

import scipy.optimize as scipy_opt
from scipy.optimize import minimize


def dual_anneal_optimizer(objective, x_guess):
    bounds = [(1e-1, 1e1)] * len(x_guess)
    bounds[1] = (1e-1, 2)
    res = scipy_opt.dual_annealing(objective,
                                   [(1e-1, 1e1)] * len(x_guess),
                                   x0=x_guess)
    est_parameters = res.x
    return est_parameters, res.fun


def diff_evo_optimizer(objective, x_guess):
    bounds = [(1e-1, 1e1)] * len(x_guess)
    bounds[1] = (1e-1, 2)
    res = scipy_opt.differential_evolution(objective,
                                           bounds)
    est_parameters = res.x
    return est_parameters, res.fun
