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

sns.set()
sns.set(font_scale=1.6)

class REEPS:
    """REEPS  (Rare earth extraction parameter searcher)
    Takes in experimental data
    Returns parameters for GEM
    Only good for 1 rare earth and 1 extractant
    :param exp_csv_filename: (str) csv file name with experimental data
    :param phases_xml_filename: (str) xml file with parameters for equilibrium calc
    :param opt_dict: (dict) optimize info {species:{thermo_prop:guess}
    :param phase_names: (list) names of phases in xml file
    :param aq_solvent_name: (str) name of aqueous solvent in xml file
    :param extractant_name: (str) name of extractant in xml file
    :param diluant_name: (str) name of diluant in xml file
    :param complex_name: (str) name of complex in xml file
    :param rare_earth_ion_name: (str) name of rare earth ion in xml file
    :param aq_solvent_rho: (float) density of solvent (g/L)
    :param extractant_rho: (float) density of extractant (g/L)
    :param diluant_rho: (float) density of diluant (g/L)
    If no density is given, molar volume/molecular weight is used from xml
    :param objective_function: (function) function to compute objective
    By default, the objective function is log mean squared error
    of distribution ratio np.log10(re_org/re_aq)
    Function needs to take inputs:
    objective_function(predicted_dict, measured_df, **kwargs)
    **kwargs is optional
    Below is the guide for referencing predicted values
    | To access                               | Use                      |
    |-------------------------------------    |--------------------------|
    | predicted rare earth eq conc in aq      | predicted_dict['re_aq']  |
    | predicted rare earth eq conc in org     | predicted_dict['re_org'] |
    | predicted hydrogen ion conc in aq       | predicted_dict['h']      |
    | predicted extractant conc in org        | predicted_dict['z']      |
    | predicted rare earth distribution ratio | predicted_dict['re_d']   |
    For measured values, use the column names in the experimental data file
    :param optimizer: (function) function to perform optimization
    By default, the optimizer is scipy's optimize function with
    default_kwargs= {"method": 'SLSQP',
                    "bounds": [(1e-1, 1e1)*len(x_guess)],
                    "constraints": (),
                    "options": {'disp': True, 'maxiter': 1000, 'ftol': 1e-6}}
    Function needs to take inputs:
    optimizer(objective_function, x_guess, **kwargs)
    **kwargs is optional
    :param temp_xml_file_path: (str) path to temporary xml file
    default is local temp folder
    """

    def __init__(self,
                 exp_csv_filename,
                 phases_xml_filename,
                 opt_dict,
                 phase_names,
                 aq_solvent_name,
                 extractant_name,
                 diluant_name,
                 complex_name,
                 rare_earth_ion_name,
                 aq_solvent_rho=None,
                 extractant_rho=None,
                 diluant_rho=None,
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
        self._complex_name = complex_name
        self._rare_earth_ion_name = rare_earth_ion_name
        self._aq_solvent_rho = aq_solvent_rho
        self._extractant_rho = extractant_rho
        self._diluant_rho = diluant_rho
        self._objective_function = None
        self.set_objective_function(objective_function)
        self._optimizer = None
        self.set_optimizer(optimizer)
        if temp_xml_file_path is None:
            temp_xml_file_path = '{0}\\temp.xml'.format(os.getenv('TEMP'))
        self._temp_xml_file_path = temp_xml_file_path
        shutil.copyfile(phases_xml_filename, self._temp_xml_file_path)
        self._phases = ct.import_phases(phases_xml_filename, phase_names)
        self._exp_df = pd.read_csv(self._exp_csv_filename)

        self._in_moles = None

        self._aq_ind = None
        self._org_ind = None

        self.set_in_moles(feed_vol=1)
        self._predicted_dict = None
        self.update_predicted_dict()

    @staticmethod
    def log_mean_squared_error(predicted_dict, meas_df):
        meas = meas_df.values[:, 2]
        pred = predicted_dict['re_org'] / predicted_dict['re_aq']
        log_pred = np.log10(pred)
        log_meas = np.log10(meas)
        obj = np.sum((log_pred - log_meas) ** 2)
        return obj

    @staticmethod
    def slsqp_optimizer(objective, x_guess):
        optimizer_kwargs = {"method": 'SLSQP',
                            "bounds": [(1e-1, 1e1)] * len(x_guess),
                            "constraints": (),
                            "options": {'disp': True, 'maxiter': 1000, 'ftol': 1e-6}}
        res = minimize(objective, x_guess, **optimizer_kwargs)
        est_parameters = res.x
        return est_parameters

    def get_exp_csv_filename(self) -> str:
        return self._exp_csv_filename

    def set_exp_csv_filename(self, exp_csv_filename):
        self._exp_csv_filename = exp_csv_filename
        self._exp_df = pd.read_csv(self._exp_csv_filename)
        self.update_predicted_dict()
        return None

    def get_phases(self) -> list:
        return self._phases

    def set_phases(self, phases_xml_filename, phase_names):
        """Change xml and phase names
        Also runs set_in_mole to set initial moles to 1 g/L"""
        self._phases_xml_filename = phases_xml_filename
        self._phase_names = phase_names
        shutil.copyfile(phases_xml_filename, self._temp_xml_file_path)
        self._phases = ct.import_phases(phases_xml_filename, phase_names)
        self.set_in_moles(feed_vol=1)
        self.update_predicted_dict()
        return None

    def get_opt_dict(self) -> dict:
        return self._opt_dict

    def set_opt_dict(self, opt_dict):
        self._opt_dict = opt_dict
        return None

    def get_aq_solvent_name(self) -> str:
        return self._aq_solvent_name

    def set_aq_solvent_name(self, aq_solvent_name):
        self._aq_solvent_name = aq_solvent_name
        return None

    def get_extractant_name(self) -> str:
        return self._extractant_name

    def set_extractant_name(self, extractant_name):
        self._extractant_name = extractant_name
        return None

    def get_diluant_name(self) -> str:
        return self._diluant_name

    def set_diluant_name(self, diluant_name):
        self._diluant_name = diluant_name
        return None

    def get_complex_name(self) -> str:
        return self._complex_name

    def set_complex_name(self, complex_name):
        self._complex_name = complex_name
        return None

    def get_rare_earth_ion_name(self) -> str:
        return self._rare_earth_ion_name

    def set_rare_earth_ion_name(self, rare_earth_ion_name):
        self._rare_earth_ion_name = rare_earth_ion_name
        return None

    def get_aq_solvent_rho(self) -> str:
        return self._aq_solvent_rho

    def set_aq_solvent_rho(self, aq_solvent_rho):
        self._aq_solvent_rho = aq_solvent_rho
        return None

    def get_extractant_rho(self) -> str:
        return self._extractant_rho

    def set_extractant_rho(self, extractant_rho):
        self._extractant_rho = extractant_rho
        return None

    def get_diluant_rho(self) -> str:
        return self._diluant_rho

    def set_diluant_rho(self, diluant_rho):
        self._diluant_rho = diluant_rho
        return None

    def set_in_moles(self, feed_vol):
        """Function that initializes mole fractions
        :param feed_vol: (float) feed volume of mixture (g/L)"""
        phases_copy = self._phases.copy()
        exp_df = self._exp_df.copy()
        solvent_name = self._aq_solvent_name
        extractant_name = self._extractant_name
        diluant_name = self._diluant_name
        solvent_rho = self._aq_solvent_rho
        extractant_rho = self._extractant_rho
        diluant_rho = self._diluant_rho
        re_name = self._rare_earth_ion_name

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

        re_ind = phases_copy[aq_ind].species_names.index(re_name)
        re_charge = phases_copy[aq_ind].species(re_ind).charge

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
        for row in exp_df.values:
            h_plus_moles = feed_vol * row[0]
            hydroxide_ions = 0
            rare_earth_moles = feed_vol * row[6]
            chlorine_moles = re_charge * rare_earth_moles + h_plus_moles
            extractant_moles = feed_vol * row[3]
            extractant_vol = extractant_moles * extractant_mw / extractant_rho
            diluant_vol = feed_vol - extractant_vol
            diluant_moles = diluant_vol * diluant_rho / diluant_mw
            complex_moles = 0

            species_moles = [aq_phase_solvent_moles,
                             h_plus_moles,
                             hydroxide_ions,
                             chlorine_moles,
                             rare_earth_moles,
                             extractant_moles,
                             diluant_moles,
                             complex_moles,
                             ]
            in_moles_data.append(species_moles)
        self._in_moles = pd.DataFrame(in_moles_data, columns=mixed.species_names)
        self.update_predicted_dict()
        return None

    def get_in_moles(self) -> pd.DataFrame:
        return self._in_moles

    def set_objective_function(self, objective_function):
        """Set objective function. see class docstring for instructions"""
        if not callable(objective_function) \
                and objective_function not in self._built_in_obj_list:
            raise Exception(
                "objective_function must be a function "
                "or in this strings list: {0}".format(self._built_in_obj_list))
        if callable(objective_function):
            if len(signature(objective_function).parameters) < 2:
                raise Exception(
                    "objective_function must be a function "
                    "with at least 3 arguments:"
                    " f(predicted_dict, experimental_df,**kwargs)")
        if objective_function == 'Log-MSE':
            objective_function = self.log_mean_squared_error
        self._objective_function = objective_function
        return None

    def get_objective_function(self):
        return self._objective_function

    def set_optimizer(self, optimizer):
        if not callable(optimizer) \
                and optimizer not in self._built_in_opt_list:
            raise Exception(
                "optimizer must be a function "
                "or in this strings list: {0}".format(self._built_in_opt_list))
        if callable(optimizer):
            if len(signature(optimizer).parameters) < 2:
                raise Exception(
                    "optimizer must be a function "
                    "with at least 2 arguments: "
                    "f(objective_func,x_guess,**kwargs)")
        if optimizer == 'SLSQP':
            optimizer = self.slsqp_optimizer
        self._optimizer = optimizer
        return None

    def get_optimizer(self):
        return self._optimizer

    def update_predicted_dict(self, phases_xml_filename=None):
        if phases_xml_filename is None:
            phases_xml_filename = self._phases_xml_filename
        phase_names = self._phase_names
        aq_ind = self._aq_ind
        org_ind = self._org_ind
        complex_name = self._complex_name
        extractant_name = self._extractant_name
        rare_earth_ion_name = self._rare_earth_ion_name
        in_moles = self._in_moles

        phases_copy = ct.import_phases(phases_xml_filename, phase_names)
        mix = ct.Mixture(phases_copy)
        predicted_dict = {"re_aq": [],
                          "re_org": [],
                          "h": [],
                          "z": []
                          }

        for row in in_moles.values:
            mix.species_moles = row
            mix.equilibrate('TP', log_level=0)
            re_org = mix.species_moles[mix.species_index(
                org_ind, complex_name)]
            re_aq = mix.species_moles[mix.species_index(
                aq_ind, rare_earth_ion_name)]
            hydrogen_ions = mix.species_moles[mix.species_index(aq_ind, 'H+')]
            extractant = mix.species_moles[mix.species_index(
                org_ind, extractant_name)]
            predicted_dict['re_aq'].append(re_aq)
            predicted_dict['re_org'].append(re_org)
            predicted_dict['h'].append(hydrogen_ions)
            predicted_dict['z'].append(extractant)
        predicted_dict['re_aq'] = np.array(predicted_dict['re_aq'])
        predicted_dict['re_org'] = np.array(predicted_dict['re_org'])
        predicted_dict['h'] = np.array(predicted_dict['h'])
        predicted_dict['z'] = np.array(predicted_dict['z'])

        self._predicted_dict = predicted_dict
        return None

    def get_predicted_dict(self):
        return self._predicted_dict

    def _internal_objective(self, x, kwargs=None):
        """default Log mean squared error between measured and predicted data
        :param x: (list) thermo properties varied to minimize LMSE
        :param kwargs: (list) arguments for objective_function
        """
        temp_xml_file_path = self._temp_xml_file_path
        exp_df = self._exp_df
        objective_function = self._objective_function
        opt_dict = copy.deepcopy(self._opt_dict)
        i = 0
        for species_name in opt_dict.keys():
            for _ in opt_dict[species_name].keys():
                i += 1
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
                    opt_dict[species_name][thermo_prop] *= x[i]
                    i += 1
            self.update_xml(opt_dict, temp_xml_file_path)

            self.update_predicted_dict(temp_xml_file_path)
            predicted_dict = self.get_predicted_dict()

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

    def fit(self, objective_function=None, optimizer=None, objective_kwargs=None, optimizer_kwargs=None) -> dict:
        """Fits experimental to modeled data by minimizing objective function
        Returns estimated complex enthalpy in J/mol
        :param objective_function: (function) function to compute objective
        :param optimizer: (function) function to perform optimization
        :param optimizer_kwargs: (dict) arguments for optimizer
        :param objective_kwargs: (dict) arguments for objective function
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
        self.update_predicted_dict()
        return opt_dict

    def update_xml(self,
                   info_dict,
                   phases_xml_filename=None):
        """updates xml file with info_dict
        :param info_dict: (dict) info in {species_names:{thermo_prop:val}}
        :param phases_xml_filename: (str) xml filename if editing other xml
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

    # noinspection PyUnusedLocal
    def parity_plot(self, species='re_aq', save_path=None, print_r_squared=False):
        """Parity plot between measured and predicted rare earth composition"""
        phases_copy = self._phases.copy()
        mix = ct.Mixture(phases_copy)
        aq_ind = self._aq_ind
        exp_df = self._exp_df
        in_moles = self._in_moles
        rare_earth_ion_name = self._rare_earth_ion_name
        pred = []
        for row in in_moles.values:
            mix.species_moles = row
            mix.equilibrate('TP', log_level=0)
            re_aq = mix.species_moles[mix.species_index(
                aq_ind, rare_earth_ion_name)]
            pred.append(re_aq)
        pred = np.array(pred)
        meas = exp_df.values[:, 1]
        min_data = np.min([pred, meas])
        max_data = np.max([pred, meas])
        min_max_data = np.array([min_data, max_data])
        fig, ax = plt.subplots()
        re_element = ''
        n_plus = 0
        for char in self._rare_earth_ion_name:
            if char.isalpha():
                re_element = '{0}{1}'.format(re_element, char)
            else:
                n_plus += 1
        re_ion_name = '$%s^{%d+}$' % (re_element, n_plus)
        p1 = sns.scatterplot(meas, pred, color="r",
                             label="{0} eq. conc. (mol/L)".format(re_ion_name),
                             legend=False)
        p2 = sns.lineplot(min_max_data, min_max_data, color="b", label="")
        if print_r_squared:
            p1.text(min_max_data[0], min_max_data[1]*0.9, '$R^2$={0:.2f}'.format(self.r_squared()))
            plt.legend(loc='lower right')
        else:
            plt.legend()
        ax.set(xlabel='Measured', ylabel='Predicted')
        plt.show()
        if save_path is not None:
            plt.savefig(save_path, bbox_inches='tight')
        return None

    def r_squared(self):
        """r-squared value comparing measured and predicted rare earth composition"""
        phases_copy = self._phases.copy()
        mix = ct.Mixture(phases_copy)
        aq_ind = self._aq_ind
        exp_df = self._exp_df
        in_moles = self._in_moles
        rare_earth_ion_name = self._rare_earth_ion_name
        pred = []
        for row in in_moles.values:
            mix.species_moles = row
            mix.equilibrate('TP', log_level=0)
            re_aq = mix.species_moles[mix.species_index(
                aq_ind, rare_earth_ion_name)]
            pred.append(re_aq)
        predicted_y = np.array(pred)
        actual_y = exp_df.values[:, 1]
        num = sum((actual_y - predicted_y) ** 2)
        den = sum((actual_y - np.mean(actual_y)) ** 2)
        r_2 = (1 - num / den)
        return r_2
