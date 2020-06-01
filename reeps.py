import cantera as ct
import pandas as pd
import numpy as np


class REEPS:
    """REEPS  (Rare earth extraction parameter searcher)
    Takes in experimental data
    Returns parameters for GEM
    :param exp_csv_filename: (str) csv file name with experimental data
    :param param_xml_file: (str) xml file with parameters for equilibrium calc
    :param x_guess: (float) guess for multiplier variable
    :param h_guess: (float) initial guess standard enthalpy (J/kmol)
    :param phase_names: (list) names of phases in xml file
    :param aq_phase_solvent_names: (list) names of solvents in aqueous phase
    :param aq_phase_solvent_rhos: (numpy.ndarray) density of solvents in aqueous phase
    :param org_phase_solvent_names: (list) names of solvents in organic phase [extractant, diluant]
    :param org_phase_solvent_rhos: (numpy.ndarray) density of solvents in organic phase
    """

    def __init__(self,
                 exp_csv_filename,
                 param_xml_file,
                 x_guess=1,
                 h_guess=-4856609.0E3,
                 phase_names=None,
                 aq_phase_solvent_names=None,
                 aq_phase_solvent_rhos=None,
                 org_phase_solvent_names=None,
                 org_phase_solvent_rhos=None,
                 ):
        self._exp_csv_filename = exp_csv_filename
        self._x_guess = x_guess
        self._h_guess = h_guess
        if phase_names is None:
            phase_names = ['HCl_electrolyte', 'PC88A_liquid', ]
        if aq_phase_solvent_names is None:
            aq_phase_solvent_names = ['H2O(L)', ]
        if aq_phase_solvent_rhos is None:
            aq_phase_solvent_rhos = np.array([1000, ])
        if org_phase_solvent_names is None:
            org_phase_solvent_names = ['(HA)2(org_phase)', 'dodecane', ]
        if org_phase_solvent_rhos is None:
            org_phase_solvent_rhos = np.array([960, 750, ])
        self._aq_phase_solvent_names = aq_phase_solvent_names
        self._aq_phase_solvent_rhos = aq_phase_solvent_rhos
        self._org_phase_solvent_names = org_phase_solvent_names
        self._org_phase_solvent_rhos = org_phase_solvent_rhos

        self._phases = ct.import_phases(param_xml_file, phase_names)
        self._exp_df = pd.read_csv(self._exp_csv_filename)

    def get_exp_csv_filename(self) -> str:
        return self._exp_csv_filename

    def set_exp_csv_filename(self, exp_csv_filename):
        self._exp_csv_filename = exp_csv_filename
        self._exp_df = pd.read_csv(self._exp_csv_filename)
        return None

    def get_phases(self) -> list:
        return self._phases

    def set_phases(self, param_xml_file, phase_names):
        self._phases = ct.import_phases(param_xml_file, phase_names)
        return None

    def get_x_guess(self) -> float:
        return self._x_guess

    def set_x_guess(self, x_guess):
        self._x_guess = x_guess
        return None

    def get_h_guess(self) -> float:
        return self._h_guess

    def set_h_guess(self, h_guess):
        self._h_guess = h_guess
        return None

    def get_aq_phase_solvent_names(self) -> list:
        return self._aq_phase_solvent_names

    def set_aq_phase_solvent_names(self, aq_phase_solvent_names):
        self._aq_phase_solvent_names = aq_phase_solvent_names
        return None

    def get_aq_phase_solvent_rhos(self) -> list:
        return self._aq_phase_solvent_rhos

    def set_aq_phase_solvent_rhos(self, aq_phase_solvent_rhos):
        self._aq_phase_solvent_rhos = aq_phase_solvent_rhos
        return None

    def get_org_phase_solvent_names(self) -> list:
        return self._org_phase_solvent_names

    def set_org_phase_solvent_names(self, org_phase_solvent_names):
        self._org_phase_solvent_names = org_phase_solvent_names
        return None

    def get_org_phase_solvent_rhos(self) -> list:
        return self._org_phase_solvent_rhos

    def set_org_phase_solvent_rhos(self, org_phase_solvent_rhos):
        self._org_phase_solvent_rhos = org_phase_solvent_rhos
        return None

    def set_in_moles(self):
        phases_copy = self._phases.copy()
        exp_df = self._exp_df.copy()
        aq_phase_solvent_names = self._aq_phase_solvent_names
        aq_phase_solvent_rhos = self._aq_phase_solvent_rhos
        org_phase_solvent_names = self._org_phase_solvent_names
        org_phase_solvent_rhos = self._org_phase_solvent_rhos

        mixed = ct.Mixture(phases_copy)
        # phase_names = [phase.name for phase in phases_copy]  # expected structure [aq_phase, org_phase]
        # phase_indices = [mixed.phase_index(phase_name) for phase_name in phase_names]
        aq_phase_solvent_mws = []
        for name in aq_phase_solvent_names:
            aq_phase_solvent_mws.append(mixed.phase(0).molecular_weights[mixed.phase(0).species_index(name)])
        org_phase_solvent_mws = []
        for name in org_phase_solvent_names:
            org_phase_solvent_mws.append(mixed.phase(1).molecular_weights[mixed.phase(1).species_index(name)])

        in_moles = []
        feed_vol = 1.  # g/L
        aq_phase_solvent_moles = feed_vol*aq_phase_solvent_rhos[0] / aq_phase_solvent_mws[0]
        for row in exp_df.values:
            h_plus_moles = feed_vol*row[0]
            hydroxide_ions = 0
            rare_earth_moles = feed_vol*row[6]
            chlorine_moles = 3*rare_earth_moles + h_plus_moles
            extractant_moles = feed_vol*row[3]
            extractant_vol = extractant_moles*org_phase_solvent_rhos[0]/org_phase_solvent_mws[0]
            diluant_vol = feed_vol - extractant_vol
            diluant_moles= diluant_vol*org_phase_solvent_rhos[1]/org_phase_solvent_mws[1]
            complex_moles=0


            species_moles = [aq_phase_solvent_moles,
                             h_plus_moles,
                             hydroxide_ions,
                             chlorine_moles,
                             rare_earth_moles,
                             extractant_moles,
                             diluant_moles,
                             complex_moles,
                             ]
            in_moles.append(species_moles)
        in_moles_df = pd.DataFrame(in_moles, columns=mixed.species_names)
        return in_moles_df


