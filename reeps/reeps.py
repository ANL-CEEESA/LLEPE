from datetime import datetime
import cantera as ct
import pandas as pd
import numpy as np
from scipy.optimize import minimize
# noinspection PyPep8Naming
import xml.etree.ElementTree as ET
import seaborn as sns
import matplotlib.pyplot as plt
import shutil
import copy
from inspect import signature
import os
import re
import pkg_resources

sns.set()
sns.set(font_scale=1.6)


class REEPS:
    r"""
    Rare earth elements (REE or RE) Takes in experimental data
    Returns parameters for GEM

    .. note::

        The order in which the REEs appear in the csv file must be the same
        order as they appear in the xml, complex_names and
        rare_earth_ion_names.

        For example, say in exp_csv_filename's csv, RE_1 is Nd RE_2 is Pr,
        and

        .. code-block:: python

            aq_solvent_name = 'H2O(L)'
            extractant_name = '(HA)2(org)'
            diluent_name = 'dodecane'

        Then:

        The csvs column ordering must be:

        [h_i, h_eq, z_i, z_eq, Nd_aq_i, Nd_aq_eq, Nd_d_eq,
        Pr_aq_i, Pr_aq_eq, Pr_d_eq]

        The aqueous speciesArray must be
        "H2O(L) H+ OH- Cl- Nd+++ Pr+++"

        The organic speciesArray must be
        "(HA)2(org) dodecane Nd(H(A)2)3(org) Pr(H(A)2)3(org)"

        .. code-block:: python

            complex_names = ['Nd(H(A)2)3(org)', 'Pr(H(A)2)3(org)']
            rare_earth_ion_names = ['Nd+++', 'Pr+++']


    :param exp_csv_filename: (str) csv file name with experimental data

        In the .csv file, the rows are different experiments and
        columns are the measured quantities.

        The ordering of the columns needs to be:

        [h_i, h_eq, z_i, z_eq,
        {RE_1}_aq_i, {RE_1}_aq_eq, {RE_1}_d_eq,
        {RE_2}_aq_i, {RE_2}_aq_eq, {RE_2}_d_eq,...
        {RE_N}_aq_i, {RE_N}_aq_eq, {RE_N}_d_eq]

        Naming does not matter, just the order.

        Where {RE_1}-{RE_N} are the rare earth element names of interest
        i.e. Nd, Pr, La, etc.

        Below is an explanation of the columns.

        +-------+------------+------------------------------------------+
        | Index |   Column   |                  Meaning                 |
        +=======+============+==========================================+
        | 0     | h_i        | Initial Concentration of                 |
        |       |            | H+ ions (mol/L)                          |
        +-------+------------+------------------------------------------+
        | 1     | h_eq       | Equilibrium concentration of             |
        |       |            | H+ ions (mol/L)                          |
        +-------+------------+------------------------------------------+
        | 2     | z_i        | Initial concentration of                 |
        |       |            | extractant (mol/L)                       |
        +-------+------------+------------------------------------------+
        | 3     | z_eq       | Equilibrium concentration of             |
        |       |            | extractant (mol/L)                       |
        +-------+------------+------------------------------------------+
        | 4     | {RE}_aq_i  | Initial concentration of RE ions (mol/L) |
        +-------+------------+------------------------------------------+
        | 5     | {RE}_aq_eq | Equilibrium concentration of RE ions     |
        |       |            | in aqueous phase (mol/L)                 |
        +-------+------------+------------------------------------------+
        | 6     | {RE}_d_eq  | Equilibrium Ratio between amount of      |
        |       |            | RE atoms in organic to aqueous           |
        +-------+------------+------------------------------------------+
    :param phases_xml_filename: (str) xml file with parameters
        for equilibrium calc

        Would recommend copying and modifying xmls located in data/xmls
        or in Cantera's "data" folder

        speciesArray fields need specific ordering.

        In aqueous phase: aq_solvent_name, H+, OH-, Cl-, RE_1, RE_2, ..., RE_N

        For aqueous phase, RE_1-RE_N represent RE ion names i.e. Nd+++, Pr+++

        In organic phase : extractant_name, diluant_name, RE_1, RE_2, ..., RE_N

        For organic phase, RE_1-RE_N represent RE complex names
        i.e. Nd(H(A)2)3(org), Pr(H(A)2)3(org)

    :param phase_names: (list) names of phases in xml file

        Found in the xml file under <phase ... id={phase_name}>

    :param aq_solvent_name: (str) name of aqueous solvent in xml file
    :param extractant_name: (str) name of extractant in xml file
    :param diluant_name: (str) name of diluant in xml file
    :param complex_names: (list) names of complexes in xml file.

        Ensure the ordering is correct
    :param rare_earth_ion_names: (list) names of rare earth ions in xml file

        Ensure the ordering is correct
    :param re_species_list: (list) names of rare earth elements.

        If ``None``, re_species_list will be rare_earth_ion_names without '+'
            i.e. 'Nd+++'->'Nd'

        Ensure the ordering is correct
    :param aq_solvent_rho: (float) density of solvent (g/L)

        If ``None``, molar volume/molecular weight is used from xml
    :param extractant_rho: (float) density of extractant (g/L)

        If ``None``, molar volume/molecular weight is used from xml
    :param diluant_rho: (float) density of diluant (g/L)

        If ``None``, molar volume/molecular weight is used from xml
    :param opt_dict: (dict) dictionary containing info about which
        species parameters are updated to fit model to experimental data

        Should have the format as below

        .. code-block:: python

            opt_dict = {"species1":
                                   {"parameter1": "guess1",
                                   "parameter2": "guess2",
                                    ...
                                    "parameterN": "guessN"}
                       "species2":
                                     {"parameter1": "guess1",
                                     "parameter2": "guess2",
                                     ...
                                     "parameterN": "guessN"}
                       ...
                       "speciesN":
                                    {"parameter1": "guess1",
                                    "parameter2": "guess2",
                                    ...
                                    "parameterN": "guessN"}
                                     }
    :param objective_function: (function or str) function to compute objective

        By default, the objective function is log mean squared error
        of distribution ratio

        .. code-block:: python

            np.sum((np.log10(d_pred)-np.log10(d_meas))^2)

        Function needs to take inputs:

        .. code-block:: python

            objective_function(predicted_dict, measured_df, kwargs)

        ``kwargs`` is optional

        Function needs to return: (float) value computed by objective function

        Below is the guide for referencing predicted values

        +---------------------------+--------------------------------+
        | To access                 | Use                            |
        +===========================+================================+
        | hydrogen ion conc in aq   | predicted_dict['h_eq']         |
        +---------------------------+--------------------------------+
        | extractant conc in org    | predicted_dict['z_eq']         |
        +---------------------------+--------------------------------+
        | RE ion eq conc in aq      | predicted_dict['{RE}_aq_eq']   |
        +---------------------------+--------------------------------+
        | RE complex eq conc in org | predicted_dict['{RE}_org_eq']  |
        +---------------------------+--------------------------------+
        | RE distribution ratio     | predicted_dict['{RE}_d_eq']    |
        +---------------------------+--------------------------------+

        Replace "{RE}" with rare earth element i.e. Nd, La, etc.

        For measured values, use the same names, but
        replace ``predicted_dict`` with ``measured_df``
    :param optimizer: (function or str) function to perform optimization

        .. note::

            The optimized variables are not directly the species parameters,
            but instead are first multiplied by the initial guess before
            sending becoming the species parameters.

            For example, say

            .. code-block:: python

                opt_dict = {'Nd(H(A)2)3(org):'h0':-4.7e6}

            If the bounds on h0 need to be [-4.7e7,-4.7e5], then
            divide the bounds by the guess and get

            .. code-block:: python

                "bounds": [(1e-1, 1e1)]

            Though fit() returns a structure identical to opt_dict with correct
            scaled values, in case bounds and constraints are used, you must
            note the optimized x's are first multiplied by the initial guess
            before written to the xml.


        By default, the optimizer is scipy's optimize function with

        .. code-block:: python

            default_kwargs= {"method": 'SLSQP',
                             "bounds": [(1e-1, 1e1)] * len(x_guess),
                             "constraints": (),
                             "options": {'disp': True,
                                         'maxiter': 1000,
                                         'ftol': 1e-6}}

        Function needs to take inputs:
        ``optimizer(objective_function, x_guess, kwargs)``

        ``kwargs`` is optional

        Function needs to return: (np.ndarray) Optimized parameters

    :param temp_xml_file_path: (str) path to temporary xml file.

        This xml file is a duplicate of the phases_xml_file name and is
        modified during the optimization process to avoid changing the original
        xml file

        default is local temp folder
    """

    def __init__(self,
                 exp_csv_filename,
                 phases_xml_filename,
                 phase_names,
                 aq_solvent_name,
                 extractant_name,
                 diluant_name,
                 complex_names,
                 rare_earth_ion_names,
                 re_species_list=None,
                 aq_solvent_rho=None,
                 extractant_rho=None,
                 diluant_rho=None,
                 opt_dict=None,
                 objective_function='Log-MSE',
                 optimizer='SLSQP',
                 temp_xml_file_path=None
                 ):
        self._built_in_obj_list = ['Log-MSE']
        self._built_in_opt_list = ['SLSQP']
        self._exp_csv_filename = exp_csv_filename
        self._phases_xml_filename = phases_xml_filename
        self._opt_dict = opt_dict
        self._phase_names = phase_names
        self._aq_solvent_name = aq_solvent_name
        self._extractant_name = extractant_name
        self._diluant_name = diluant_name
        self._complex_names = complex_names
        self._rare_earth_ion_names = rare_earth_ion_names
        self._aq_solvent_rho = aq_solvent_rho
        self._extractant_rho = extractant_rho
        self._diluant_rho = diluant_rho
        self._objective_function = None
        self.set_objective_function(objective_function)
        self._optimizer = None
        self._re_species_list = re_species_list
        self.set_optimizer(optimizer)
        if temp_xml_file_path is None:
            temp_xml_file_path = r'{0}/temp.xml'.format(os.getenv('TEMP'))
        self._temp_xml_file_path = temp_xml_file_path
        # Try and except for adding package data to path.
        # This only works for sdist, not bdist
        # If bdist is needed, research "manifest.in" python setup files
        try:
            shutil.copyfile(self._phases_xml_filename,
                            self._temp_xml_file_path)
            self._phases = ct.import_phases(self._phases_xml_filename,
                                            phase_names)
        except FileNotFoundError:
            self._phases_xml_filename = \
                pkg_resources.resource_filename('reeps',
                                                r'..\data\xmls\{0}'.format(
                                                    phases_xml_filename))
            shutil.copyfile(self._phases_xml_filename,
                            self._temp_xml_file_path)
            self._phases = ct.import_phases(self._phases_xml_filename,
                                            phase_names)
        try:
            self._exp_df = pd.read_csv(self._exp_csv_filename)
        except FileNotFoundError:
            self._exp_csv_filename = pkg_resources.resource_filename(
                'reeps', r'..\data\csvs\{0}'.format(self._exp_csv_filename))
            self._exp_df = pd.read_csv(self._exp_csv_filename)

        self._exp_df_columns = ['h_i', 'h_eq', 'z_i', 'z_eq']
        if self._re_species_list is None:
            self._re_species_list = []
            for name in self._rare_earth_ion_names:
                species = name.replace('+', '')
                self._re_species_list.append(species)
        for species in self._re_species_list:
            self._exp_df_columns.append('{0}_aq_i'.format(species))
            self._exp_df_columns.append('{0}_aq_eq'.format(species))
            self._exp_df_columns.append('{0}_d_eq'.format(species))
        self._exp_df.columns = self._exp_df_columns

        self._in_moles = None

        self._aq_ind = None
        self._org_ind = None
        self._re_charges = None

        self.set_in_moles(feed_vol=1)
        self._predicted_dict = None
        self.update_predicted_dict()

    @staticmethod
    def scipy_minimize(objective, x_guess, optimizer_kwargs=None):
        """ The default optimizer for REEPS

        Uses scipy.minimize

        By default, options are

        .. code-block:: python

            default_kwargs= {"method": 'SLSQP',
                            "bounds": [(1e-1, 1e1)]*len(x_guess),
                            "constraints": (),
                            "options": {'disp': True,
                                        'maxiter': 1000,
                                        'ftol': 1e-6}}

        :param objective: (func) the objective function
        :param x_guess: (np.ndarray) the initial guess (always 1)
        :param optimizer_kwargs: (dict) dictionary of options for minimize
        :returns: (np.ndarray) Optimized parameters
        """
        if optimizer_kwargs is None:
            optimizer_kwargs = {"method": 'SLSQP',
                                "bounds": [(1e-1, 1e1)] * len(x_guess),
                                "constraints": (),
                                "options": {'disp': True,
                                            'maxiter': 1000,
                                            'ftol': 1e-6}}
        res = minimize(objective, x_guess, **optimizer_kwargs)
        est_parameters = res.x
        return est_parameters

    def log_mean_squared_error(self, predicted_dict, meas_df):
        """Default objective function for REEPS

        Returns the log mean squared error of
        predicted distribution ratios  (d=n_org/n_aq)
        to measured d.

        np.sum((np.log10(d_pred)-np.log10(d_meas))\**2)

        :param predicted_dict: (dict) contains predicted data
        :param meas_df: (pd.DataFrame) contains experimental data
        :return: (float) log mean squared error between predicted and measured
        """
        meas = np.concatenate([meas_df['{0}_d_eq'.format(species)].values
                               for species in self._re_species_list])
        pred = np.concatenate([
            predicted_dict['{0}_d_eq'.format(species)]
            for species in self._re_species_list])
        log_pred = np.log10(pred)
        log_meas = np.log10(meas)
        obj = np.sum((log_pred - log_meas) ** 2)
        return obj

    def get_exp_df(self) -> pd.DataFrame:
        """Returns the experimental DataFrame

        :return: (pd.DataFrame) Experimental data
        """
        return self._exp_df

    def set_exp_df(self, exp_csv_filename):
        """Changes the experimental DataFrame to input exp_csv_filename data
        and renames columns to internal REEPS names


        h_i, h_eq, z_i, z_eq, {RE}_aq_i, {RE}_aq_eq, {RE}_d

        See class docstring on "exp_csv_filename" for further explanations.

        :param exp_csv_filename: (str) file name/path for experimental data csv
        """
        self._exp_csv_filename = exp_csv_filename
        try:
            self._exp_df = pd.read_csv(self._exp_csv_filename)
        except FileNotFoundError:
            self._exp_csv_filename = pkg_resources.resource_filename(
                'reeps', r'..\data\csvs\{0}'.format(self._exp_csv_filename))
            self._exp_df = pd.read_csv(self._exp_csv_filename)
        self._exp_df_columns = ['h_i', 'h_eq', 'z_i', 'z_eq']
        if self._re_species_list is None:
            self._re_species_list = []
            for name in self._rare_earth_ion_names:
                species = name.replace('+', '')
                self._re_species_list.append(species)
        for species in self._re_species_list:
            self._exp_df_columns.append('{0}_aq_i'.format(species))
            self._exp_df_columns.append('{0}_aq_eq'.format(species))
            self._exp_df_columns.append('{0}_d_eq'.format(species))
        self._exp_df.columns = self._exp_df_columns
        self.set_in_moles(feed_vol=1)
        self.update_predicted_dict()
        return None

    def get_phases(self) -> list:
        """
        Returns the list of Cantera solutions

        :return: (list) list of Cantera solutions/phases
        """
        return self._phases

    def set_phases(self, phases_xml_filename, phase_names):
        """Change list of Cantera solutions by inputting
        new xml file name and phase names

        Also runs set_in_moles to set feed volume to 1 L

        :param phases_xml_filename: (str) xml file with parameters
            for equilibrium calc
        :param phase_names: (list) names of phases in xml file
        """
        self._phases_xml_filename = phases_xml_filename
        self._phase_names = phase_names
        # Try and except for adding package data to path.
        # This only works for sdist, not bdist
        # If bdist is needed, research "manifest.in" python setup files
        try:
            shutil.copyfile(self._phases_xml_filename,
                            self._temp_xml_file_path)
            self._phases = ct.import_phases(self._phases_xml_filename,
                                            phase_names)
        except FileNotFoundError:
            self._phases_xml_filename = \
                pkg_resources.resource_filename('reeps',
                                                r'..\data\xmls\{0}'.format(
                                                    phases_xml_filename))
            shutil.copyfile(self._phases_xml_filename,
                            self._temp_xml_file_path)
            self._phases = ct.import_phases(self._phases_xml_filename,
                                            phase_names)
        self.set_in_moles(feed_vol=1)
        self.update_predicted_dict()
        return None

    def get_opt_dict(self) -> dict:
        """
        Returns the dictionary containing optimization information

        :return: (dict) dictionary containing info about which
            species parameters are updated to fit model to experimental data
        """
        return self._opt_dict

    def set_opt_dict(self, opt_dict):
        """
        Change the dictionary to input opt_dict.

        opt_dict specifies species parameters to be updated to
        fit model to data

        See class docstring on "opt_dict" for more information.

        :param opt_dict: (dict) dictionary containing info about which
            species parameters are updated to fit model to experimental data
        """

        self._opt_dict = opt_dict
        return None

    def get_aq_solvent_name(self) -> str:
        """Returns aq_solvent_name

        :return: aq_solvent_name: (str) name of aqueous solvent in xml file
        """
        return self._aq_solvent_name

    def set_aq_solvent_name(self, aq_solvent_name):
        """ Change aq_solvent_name to input aq_solvent_name

        :param aq_solvent_name: (str) name of aqueous solvent in xml file
        """
        self._aq_solvent_name = aq_solvent_name
        return None

    def get_extractant_name(self) -> str:
        """Returns extractant name

        :return: extractant_name: (str) name of extractant in xml file
        """
        return self._extractant_name

    def set_extractant_name(self, extractant_name):
        """
        Change extractant_name to input extractant_name
        :param extractant_name: (str) name of extractant in xml file
        """
        self._extractant_name = extractant_name
        return None

    def get_diluant_name(self) -> str:
        """ Returns diluant name
        :return:  diluant_name: (str) name of diluant in xml file
        """
        return self._diluant_name

    def set_diluant_name(self, diluant_name):
        """
        Change diluant_name to input diluant_name

        :param diluant_name: (str) name of diluant in xml file
        """
        self._diluant_name = diluant_name
        return None

    def get_complex_names(self) -> list:
        """Returns list of complex names

        :return: complex_names: (list) names of complexes in xml file.
        """
        return self._complex_names

    def set_complex_names(self, complex_names):
        """Change complex names list to input complex_names

        :param complex_names: (list) names of complexes in xml file.
        """
        self._complex_names = complex_names
        return None

    def get_rare_earth_ion_names(self) -> list:
        """Returns list of rare earth ion names

        :return: rare_earth_ion_names: (list) names of rare earth ions in
            xml file
        """
        return self._rare_earth_ion_names

    def set_rare_earth_ion_names(self, rare_earth_ion_names):
        """Change list of rare earth ion names to input
            rare_earth_ion_names

        :param rare_earth_ion_names: (list) names of rare earth ions in
            xml file
        """
        self._rare_earth_ion_names = rare_earth_ion_names
        return None

    def get_re_species_list(self) -> list:
        """Returns list of rare earth element names

        :return: re_species_list: (list) names of rare earth elements in
            xml file
        """
        return self._re_species_list

    def set_re_species_list(self, re_species_list):
        """Change list of rare earth ion names to input
            rare_earth_ion_names

        :param re_species_list: (list) names of rare earth elements in
            xml file
        """
        self._re_species_list = re_species_list
        return None

    def get_aq_solvent_rho(self) -> str:
        """Returns aqueous solvent density (g/L)

        :return: aq_solvent_rho: (float) density of aqueous solvent
        """
        return self._aq_solvent_rho

    def set_aq_solvent_rho(self, aq_solvent_rho):
        """Changes aqueous solvent density (g/L) to input aq_solvent_rho

        :param aq_solvent_rho: (float) density of aqueous solvent
        """
        self._aq_solvent_rho = aq_solvent_rho
        return None

    def get_extractant_rho(self) -> str:
        """Returns extractant density (g/L)

        :return: extractant_rho: (float) density of extractant
        """
        return self._extractant_rho

    def set_extractant_rho(self, extractant_rho):
        """Changes extractant density (g/L) to input extractant_rho

        :param extractant_rho: (float) density of extractant
        """
        self._extractant_rho = extractant_rho
        return None

    def get_diluant_rho(self) -> str:
        """Returns diluant density (g/L)

        :return: diluant_rho: (float) density of diluant
        """
        return self._diluant_rho

    def set_diluant_rho(self, diluant_rho):
        """Changes diluant density (g/L) to input diluant_rho

        :param diluant_rho: (float) density of diluant
        """
        self._diluant_rho = diluant_rho
        return None

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
        re_names = self._rare_earth_ion_names
        re_species_list = self._re_species_list

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
        diluant_ind = phases_copy[org_ind].species_names.index(diluant_name)

        re_ind_list = [phases_copy[aq_ind].species_names.index(re_name)
                       for re_name in re_names]
        re_charges = np.array([phases_copy[aq_ind].species(re_ind).charge
                               for re_ind in re_ind_list])
        self._re_charges = re_charges

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
            rare_earth_moles = np.array([feed_vol * row[
                '{0}_aq_i'.format(re_species)]
                                         for re_species in re_species_list])
            re_charge_sum = np.sum(re_charges * rare_earth_moles)
            chlorine_moles = re_charge_sum + h_plus_moles
            extractant_moles = feed_vol * row['z_i']
            extractant_vol = extractant_moles * extractant_mw / extractant_rho
            diluant_vol = feed_vol - extractant_vol
            diluant_moles = diluant_vol * diluant_rho / diluant_mw
            complex_moles = np.zeros(len(re_species_list))

            species_moles_aq = [aq_phase_solvent_moles,
                                h_plus_moles,
                                hydroxide_ions,
                                chlorine_moles]
            species_moles_aq.extend(list(rare_earth_moles))
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

    def get_in_moles(self) -> pd.DataFrame:
        """Returns the in_moles DataFrame which contains the initial mole
        fractions of each species for each experiment

        :return: in_moles: (pd.DataFrame) DataFrame with initial mole fractions
        """
        return self._in_moles

    def set_objective_function(self, objective_function):
        """Change objective function to input objective_function.

         See class docstring on "objective_function" for instructions

        :param objective_function: (func) Objective function to quantify
            error between model and experimental data
        """
        if not callable(objective_function) \
                and objective_function not in self._built_in_obj_list:
            raise Exception(
                "objective_function must be a function "
                "or in this strings list: {0}".format(
                    self._built_in_obj_list))
        if callable(objective_function):
            if len(signature(objective_function).parameters) < 2:
                raise Exception(
                    "objective_function must be a function "
                    "with at least 3 arguments:"
                    " f(predicted_dict, experimental_df, kwargs)")
        if objective_function == 'Log-MSE':
            objective_function = self.log_mean_squared_error
        self._objective_function = objective_function
        return None

    def get_objective_function(self):
        """Returns objective function

        :return: objective_function: (func) Objective function to quantify
            error between model and experimental data
        """
        return self._objective_function

    def set_optimizer(self, optimizer):
        """Change optimizer function to input optimizer.

         See class docstring on "optimizer" for instructions

        :param optimizer: (func) Optimizer function to minimize objective
            function
        """
        if not callable(optimizer) \
                and optimizer not in self._built_in_opt_list:
            raise Exception(
                "optimizer must be a function "
                "or in this strings list: {0}".format(
                    self._built_in_opt_list))
        if callable(optimizer):
            if len(signature(optimizer).parameters) < 2:
                raise Exception(
                    "optimizer must be a function "
                    "with at least 2 arguments: "
                    "f(objective_func,x_guess, kwargs)")
        if optimizer == 'SLSQP':
            optimizer = self.scipy_minimize
        self._optimizer = optimizer
        return None

    def get_optimizer(self):
        """Returns objective function

        :return: optimizer: (func) Optimizer function to minimize objective
            function
        """
        return self._optimizer

    def get_temp_xml_file_path(self):
        """Returns path to temporary xml file.

        This xml file is a duplicate of the phases_xml_file name and is
        modified during the optimization process to avoid changing the original
        xml file.

        :return: temp_xml_file_path: (str) path to temporary xml file.
        """
        return self._temp_xml_file_path

    def set_temp_xml_file_path(self, temp_xml_file_path):
        """Changes temporary xml file path to input temp_xml_file_path.

        This xml file is a duplicate of the phases_xml_file name and is
        modified during the optimization process to avoid changing the original
        xml file.

        :param temp_xml_file_path: (str) path to temporary xml file.
        """
        self._temp_xml_file_path = temp_xml_file_path
        return None

    def update_predicted_dict(self,
                              phases_xml_filename=None,
                              phase_names=None):
        """Function that computes the predicted equilibrium concentrations
        the fed phases_xml_filename parameters predicts given the initial
        mole fractions set by in_moles()

        :param phases_xml_filename: (str)xml file with parameters
            for equilibrium calc. If ``None``, the
            current phases_xml_filename is used.
        :param phase_names: (list) names of phases in xml file.
            If ``None``, the current phases_names is used.
        """
        if phases_xml_filename is None:
            phases_xml_filename = self._phases_xml_filename
        if phase_names is None:
            phase_names = self._phase_names
        aq_ind = self._aq_ind
        org_ind = self._org_ind
        complex_names = self._complex_names
        extractant_name = self._extractant_name
        rare_earth_ion_names = self._rare_earth_ion_names
        in_moles = self._in_moles
        re_species_list = self._re_species_list

        phases_copy = ct.import_phases(phases_xml_filename, phase_names)
        mix = ct.Mixture(phases_copy)
        key_names = ['h_eq', 'z_eq']
        for re_species in re_species_list:
            key_names.append('{0}_aq_eq'.format(re_species))
            key_names.append('{0}_org_eq'.format(re_species))
            key_names.append('{0}_d_eq'.format(re_species))

        predicted_dict = {'{0}'.format(key_name): []
                          for key_name in key_names}

        for row in in_moles.values:
            mix.species_moles = row
            mix.equilibrate('TP', log_level=0)
            re_org_array = np.array([mix.species_moles[mix.species_index(
                org_ind, complex_name)] for complex_name in complex_names])
            re_aq_array = np.array([mix.species_moles[mix.species_index(
                aq_ind, re_ion_name)] for re_ion_name in rare_earth_ion_names])
            d_array = re_org_array / re_aq_array
            hydrogen_ions = mix.species_moles[mix.species_index(aq_ind, 'H+')]
            extractant = mix.species_moles[mix.species_index(
                org_ind, extractant_name)]
            for index, re_species in enumerate(re_species_list):
                predicted_dict['{0}_aq_eq'.format(
                    re_species)].append(re_aq_array[index])
                predicted_dict['{0}_org_eq'.format(
                    re_species)].append(re_org_array[index])
                predicted_dict['{0}_d_eq'.format(
                    re_species)].append(d_array[index])
            predicted_dict['h_eq'].append(hydrogen_ions)
            predicted_dict['z_eq'].append(extractant)
        for key, value in predicted_dict.items():
            predicted_dict[key] = np.array(value)
        self._predicted_dict = predicted_dict
        return None

    def get_predicted_dict(self):
        """Returns predicted dictionary of species concentrations
        that xml parameters predicts given current in_moles

        :return: predicted_dict: (dict) dictionary of species concentrations
        """
        return self._predicted_dict

    def _internal_objective(self, x, kwargs=None):
        """
        Internal objective function. Uses objective function to compute value
        If the optimizer requires vectorized variables ie pso, this function
        takes care of it

        :param x: (list) thermo properties varied to minimize objective func
        :param kwargs: (list) arguments for objective_function
        """
        temp_xml_file_path = self._temp_xml_file_path
        exp_df = self._exp_df
        objective_function = self._objective_function
        opt_dict = copy.deepcopy(self._opt_dict)
        x = np.array(x)

        if len(x.shape) == 1:
            xs = np.array([x])
            vectorized_x = False
        else:
            vectorized_x = True
            xs = x
        objective_values = []
        for x in xs:
            i = 0
            for species_name in opt_dict.keys():
                for thermo_prop in opt_dict[species_name].keys():
                    if not np.isnan(
                            x[i]):  # if nan, do not update xml with nan
                        opt_dict[species_name][thermo_prop] *= x[i]
                    i += 1
            self.update_xml(opt_dict, temp_xml_file_path)

            self.update_predicted_dict(temp_xml_file_path)
            predicted_dict = self.get_predicted_dict()
            self.update_predicted_dict()

            if kwargs is None:
                # noinspection PyCallingNonCallable
                obj = objective_function(predicted_dict, exp_df)
            else:
                # noinspection PyCallingNonCallable
                obj = objective_function(predicted_dict, exp_df, **kwargs)
            objective_values.append(obj)
        if vectorized_x:
            objective_values = np.array(objective_values)
        else:
            objective_values = objective_values[0]
        return objective_values

    def fit(self,
            objective_function=None,
            optimizer=None,
            objective_kwargs=None,
            optimizer_kwargs=None) -> dict:
        """Fits experimental to modeled data by minimizing objective function
        with optimizer. Returns dictionary with opt_dict structure

        :param objective_function: (function) function to compute objective
            If 'None', last set objective or default function is used
        :param optimizer: (function) function to perform optimization
            If 'None', last set optimizer or default is used
        :param optimizer_kwargs: (dict) optional arguments for optimizer
        :param objective_kwargs: (dict) optional arguments
            for objective function
        :returns opt_dict: (dict) optimized opt_dict. Has identical structure
            as opt_dict
        """
        if objective_function is not None:
            self.set_objective_function(objective_function)
        if optimizer is not None:
            self.set_optimizer(optimizer)

        def objective(x):
            return self._internal_objective(x, objective_kwargs)

        optimizer = self._optimizer
        opt_dict = copy.deepcopy(self._opt_dict)
        i = 0
        for species_name in opt_dict.keys():
            for _ in opt_dict[species_name].keys():
                i += 1
        x_guess = np.ones(i)

        if optimizer_kwargs is None:
            # noinspection PyCallingNonCallable
            est_parameters = optimizer(objective, x_guess)
        else:
            # noinspection PyCallingNonCallable
            est_parameters = optimizer(objective, x_guess, **optimizer_kwargs)

        i = 0
        for species_name in opt_dict.keys():
            for thermo_prop in opt_dict[species_name].keys():
                opt_dict[species_name][thermo_prop] *= est_parameters[i]
                i += 1
        return opt_dict

    def update_xml(self,
                   info_dict,
                   phases_xml_filename=None):
        """updates xml file with info_dict

        :param info_dict: (dict) info in {species_names:{thermo_prop:val}}
            Requires an identical structure to opt_dict
        :param phases_xml_filename: (str) xml filename if editing other xml
            If ``None``, the current xml will be modified and the internal
            Cantera phases will be refreshed to the new values.
        """
        if phases_xml_filename is None:
            phases_xml_filename = self._phases_xml_filename

        tree = ET.parse(phases_xml_filename)
        root = tree.getroot()
        # Update xml file
        for species_name in info_dict.keys():
            for thermo_prop in info_dict[species_name].keys():
                for species in root.iter('species'):
                    if species.attrib['name'] == species_name:
                        for changed_prop in species.iter(thermo_prop):
                            changed_prop.text = str(
                                info_dict[species_name][thermo_prop])
                            now = datetime.now()
                            changed_prop.set('updated',
                                             'Updated at {0}:{1} {2}-{3}-{4}'
                                             .format(now.hour, now.minute,
                                                     now.month, now.day,
                                                     now.year))

        tree.write(phases_xml_filename)
        if phases_xml_filename == self._phases_xml_filename:
            self.set_phases(self._phases_xml_filename, self._phase_names)
        return None

    def parity_plot(self,
                    compared_value=None,
                    color_axis=None,
                    plot_title=None,
                    save_path=None,
                    print_r_squared=False):
        """
        Parity plot between measured and predicted compared_value.
        Default compared value is {RE_1}_aq_eq

        :param compared_value: (str) Quantity to compare predicted and
            experimental data. Can be any column containing "eq" in exp_df i.e.
            h_eq, z_eq, {RE}_d_eq, etc.
        :param plot_title: (str or boolean)

            If None (default): Plot title will be generated from compared_value
                Recommend to just explore. If h_eq, plot_title is
                "H^+ eq conc".

            If str: Plot title will be plot_title string

            If "False": No plot title
        :param color_axis: (dict)
        :param save_path: (str) save path for parity plot
        :param print_r_squared: (boolean) To plot or not to plot r-squared
            value. Prints 2 places past decimal
        """
        exp_df = self.get_exp_df()
        predicted_dict = self.get_predicted_dict()
        re_species_list = self._re_species_list
        extractant_name = self.get_extractant_name()
        re_charges = self._re_charges
        if compared_value is None:
            compared_value = '{0}_aq_eq'.format(re_species_list[0])
        pred = predicted_dict[compared_value]
        meas = exp_df[compared_value].values
        min_data = np.min([pred, meas])
        max_data = np.max([pred, meas])
        min_max_data = np.array([min_data, max_data])
        name_breakdown = re.findall('[^_\W]+', compared_value)
        compared_species = name_breakdown[0]
        if compared_species == 'h':
            default_title = '$H^+$ eq. conc. (mol/L)'
        elif compared_species == 'z':
            default_title = '{0} eq. conc. (mol/L)'.format(extractant_name)
        else:
            phase = name_breakdown[1]
            if phase == 'aq':
                re_charge = re_charges[re_species_list.index(compared_species)]
                default_title = '$%s^{%d+}$ eq. conc. (mol/L)' \
                                % (compared_species, re_charge)
            elif phase == 'd':
                default_title = '{0} distribution ratio'.format(
                    compared_species)
            else:
                default_title = '{0} complex eq. conc. (mol/L)'.format(
                    compared_species)
        fig, ax = plt.subplots()
        if color_axis is None:
            sns.scatterplot(meas, pred, color="r",
                            legend=False)
        else:
            key = list(color_axis.keys())[0]
            value = list(color_axis.values())[0]
            if key == 'predicted':
                y = self.get_predicted_dict()[value]
            elif key == 'measured':
                y = self.get_exp_df()[value].values
            else:
                raise Exception('color_axis must be a dictionary with key'
                                '"predicted" or "measured"')
            y = np.array(y)
            meas = np.array(meas)
            pred = np.array(pred)

            p1 = ax.scatter(meas, pred, c=y, alpha=1, cmap='viridis')
            c_bar = fig.colorbar(p1, format='%.2f')
            # Fix next line. value is just the dictionary value.
            c_bar.set_label(value, rotation=270, labelpad=20)
        sns.lineplot(min_max_data, min_max_data, color="b", label="")

        if print_r_squared:
            ax.text(min_max_data[0],
                    min_max_data[1] * 0.9,
                    '$R^2$={0:.2f}'.format(self.r_squared(compared_value)))
            # plt.legend(loc='lower right')
        # else:
        # plt.legend()
        ax.set(xlabel='Measured', ylabel='Predicted')
        if plot_title is None:
            ax.set_title(default_title)
        elif isinstance(plot_title, str):
            ax.set_title(plot_title)
        plt.show()
        if save_path is not None:
            plt.savefig(save_path, bbox_inches='tight')
        return None

    def r_squared(self, compared_value=None):
        """r-squared value comparing measured and predicted compared value

        Closer to 1, the better the model's predictions.

        :param compared_value: (str) Quantity to compare predicted and
            experimental data. Can be any column containing "eq" in exp_df i.e.
            h_eq, z_eq, {RE}_d_eq, etc. default is {RE}_aq_eq
             """
        exp_df = self.get_exp_df()
        predicted_dict = self.get_predicted_dict()
        re_species_list = self._re_species_list
        if compared_value is None:
            compared_value = '{0}_aq_eq'.format(re_species_list[0])
        pred = predicted_dict[compared_value]
        predicted_y = np.array(pred)
        actual_y = exp_df[compared_value].values
        num = sum((actual_y - predicted_y) ** 2)
        den = sum((actual_y - np.mean(actual_y)) ** 2)
        r_2 = (1 - num / den)
        return r_2
