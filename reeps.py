import time
import cantera as ct
import pandas as pd
import numpy as np
from scipy.optimize import minimize
# noinspection PyPep8Naming
import xml.etree.ElementTree as ET
import seaborn as sns
import matplotlib.pyplot as plt

sns.set()


class REEPS:
    """REEPS  (Rare earth extraction parameter searcher)
    Takes in experimental data
    Returns parameters for GEM
    :param exp_csv_filename: (str) csv file name with experimental data
    :param phases_xml_filename: (str) xml file with parameters for equilibrium calc
    :param x_guess: (float) guess for multiplier variable
    :param h_guess: (float) initial guess standard enthalpy (J/kmol)
    :param phase_names: (list) names of phases in xml file
    :param aq_solvent_name: (str) name of aqueous solvent in xml file
    :param extractant_name: (str) name of extractant in xml file
    :param diluant_name: (str) name of diluant in xml file
    :param complex_name: (str) name of complex in xml file
    :param rare_earth_ion_name: (str) name of rare earth ion in xml file
    :param aq_solvent_rho: (float) density of solvent (g/L)
    :param extractant_rho: (float) density of extractant (g/L)
    :param diluant_rho: (float) density of extractant (g/L)
    If no density is given, molar volume/molecular weight is used from xml
    """

    def __init__(self,
                 exp_csv_filename,
                 phases_xml_filename,
                 x_guess,
                 h_guess,
                 phase_names,
                 aq_solvent_name,
                 extractant_name,
                 diluant_name,
                 complex_name,
                 rare_earth_ion_name,
                 aq_solvent_rho=None,
                 extractant_rho=None,
                 diluant_rho=None,
                 ):
        self._exp_csv_filename = exp_csv_filename
        self._phases_xml_filename = phases_xml_filename
        self._x_guess = x_guess
        self._h_guess = h_guess
        self._phase_names = phase_names
        self._aq_solvent_name = aq_solvent_name
        self._extractant_name = extractant_name
        self._diluant_name = diluant_name
        self._complex_name = complex_name
        self._rare_earth_ion_name = rare_earth_ion_name
        self._aq_solvent_rho = aq_solvent_rho
        self._extractant_rho = extractant_rho
        self._diluant_rho = diluant_rho

        self._phases = ct.import_phases(phases_xml_filename, phase_names)
        self._exp_df = pd.read_csv(self._exp_csv_filename)

        self._in_moles = None

        self._aq_ind = None
        self._org_ind = None

        self.set_in_moles(feed_vol=1)

    def get_exp_csv_filename(self) -> str:
        return self._exp_csv_filename

    def set_exp_csv_filename(self, exp_csv_filename):
        self._exp_csv_filename = exp_csv_filename
        self._exp_df = pd.read_csv(self._exp_csv_filename)
        return None

    def get_phases(self) -> list:
        return self._phases

    def set_phases(self, phases_xml_filename, phase_names):
        """Change xml and phase names
        Also runs set_in_mole to set initial moles to 1 g/L"""
        self._phases_xml_filename = phases_xml_filename
        self._phase_names = phase_names
        self._phases = ct.import_phases(phases_xml_filename, phase_names)
        self.set_in_moles(feed_vol=1)
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
        """:param feed_vol: (float) feed volume of mixture (g/L)"""
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
        return None

    def get_in_moles(self) -> pd.DataFrame:
        return self._in_moles

    def objective(self, x):
        h = x[0] * self._h_guess
        phases_copy = self._phases.copy()
        aq_ind = self._aq_ind
        org_ind = self._org_ind
        complex_name = self._complex_name
        rare_earth_ion_name = self._rare_earth_ion_name
        in_moles = self._in_moles
        exp_df = self._exp_df

        complex_index = phases_copy[org_ind].species_index(complex_name)
        complex_species = phases_copy[org_ind].species(complex_index)

        new_thermo_coeffs = complex_species.thermo.coeffs
        new_thermo_coeffs[1] = h
        complex_species.thermo = ct.ConstantCp(
            complex_species.thermo.min_temp,
            complex_species.thermo.max_temp,
            complex_species.thermo.reference_pressure,
            new_thermo_coeffs)
        phases_copy[org_ind].modify_species(complex_index, complex_species)

        mix = ct.Mixture(phases_copy)
        pred = []
        for row in in_moles.values:
            mix.species_moles = row
            mix.equilibrate('TP', log_level=0)
            re_org = mix.species_moles[mix.species_index(
                org_ind, complex_name)]
            re_aq = mix.species_moles[mix.species_index(
                aq_ind, rare_earth_ion_name)]
            pred.append(np.log10(re_org / re_aq))
        pred = np.array(pred)
        meas = np.log10(exp_df['D(m)'].values)
        obj = np.sum((pred - meas) ** 2)
        return obj

    def fit(self, kwargs) -> float:
        """Fits experimental to modeled data by estimating complex reference enthalpy
        Returns estimated complex enthalpy in J/mol
        :param kwargs: (dict) parameters and options for scipy.minimize
        """
        x_guess = self._x_guess
        h_guess = self._h_guess
        res = minimize(self.objective, x_guess, **kwargs)
        pred_complex_enthalpy = res.x[0] * h_guess / 1000  # J/mol
        return pred_complex_enthalpy

    def update_xml(self,
                   complex_enthalpy,
                   phases_xml_filename=None,
                   complex_name=None):
        """updates xml file with new complex_enthalpy"""
        if phases_xml_filename is None:
            phases_xml_filename = self._phases_xml_filename
        if complex_name is None:
            complex_name = self._complex_name

        tree = ET.parse(phases_xml_filename)
        root = tree.getroot()
        # Update xml file
        for species in root.iter('species'):
            if species.attrib['name'] is complex_name:
                for h0 in species.iter('h0'):
                    h0.text = str(complex_enthalpy)
                    h0.set('updated', 'Updated at ' + time.strftime('%X'))

        tree.write(phases_xml_filename)

    def parity_plot(self):
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
        meas = exp_df['REeq(m)'].values
        min_data = np.min([pred, meas])
        max_data = np.max([pred, meas])
        min_max_data = np.array([min_data, max_data])
        fig, ax = plt.subplots()
        sns.scatterplot(meas, pred, color="r")
        sns.lineplot(min_max_data, min_max_data, color="b")
        ax.set(xlabel='Measured X equilibrium', ylabel='Predicted X equilibrium')
        plt.show()
        return None
