import cantera as ct
import pandas as pd


class REEGEMParamFit:
    """Takes in experimental data
    Returns parameters for GEM
    :param exp_csv_file: (str) csv file containing experimental data
    :param param_xml_file: (str) xml file with parameters for equilibrium calc
    :param x_guess: (float) guess for multiplier variable
    :param h_guess: (float) initial guess standard enthalpy (J/kmol)
    :param phase_names: (list) names of phases in xml file
    """

    def __init__(self,
                 exp_csv_file,
                 param_xml_file,
                 x_guess=1,
                 h_guess=-4856609.0E3,
                 phase_names=['HCl_electrolyte', 'PC88A_liquid']
                 ):
        self.x_guess = x_guess
        self.h_guess = h_guess
        exp_df = pd.read_csv(exp_csv_file)
        phases = ct.import_phases(param_xml_file, phase_names)
