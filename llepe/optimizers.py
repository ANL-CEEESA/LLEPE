import scipy.optimize as scipy_opt
from scipy.optimize import minimize
# import skopt

def dual_anneal_optimizer(objective, x_guess):
    bounds = [(1e-1, 1e1)] * len(x_guess)
    bounds[1] = (1e-1, 2)
    res = scipy_opt.dual_annealing(objective,
                                   [(1e-1, 1e1)]*len(x_guess),
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


# def forest_lbfgsb_optimizer(objective, x_guess):
#     x_guess = list(x_guess)
#     bounds = [(1e-1, 1e1)]*len(x_guess)
#     bounds[1] = (1e-1, 2)
#     res = skopt.forest_minimize(objective,
#                                 bounds,
#                                 random_state=1,
#                                 acq_func='LCB',
#                                 n_random_starts=30,
#                                 x0=x_guess,
#                                 xi=1e-4)
#     x_guess = res.x
#     optimizer_kwargs = {"method": 'l-bfgs-b',
#                         "bounds": bounds}
#     res = minimize(objective, x_guess, **optimizer_kwargs)
#     return res.x, res.fun
