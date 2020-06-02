import cantera as ct
import pandas as pd


class REEPS:
    """REEPS  (Rare earth extraction parameter searcher)
    Takes in experimental data
    Returns parameters for GEM
    :param exp_csv_filename: (str) csv file name with experimental data
    :param param_xml_file: (str) xml file with parameters for equilibrium calc
    :param x_guess: (float) guess for multiplier variable
    :param h_guess: (float) initial guess standard enthalpy (J/kmol)
    :param phase_names: (list) names of phases in xml file
    :param aq_solvent_name: (str) name of aqueous solvent in xml file
    :param extractant_name: (str) name of extractant
    :param diluant_name: (str) name of diluant
    :param aq_solvent_rho: (float) density of solvent (g/L)
    :param extractant_rho: (float) density of extractant (g/L)
    :param diluant_rho: (float) density of extractant (g/L)
    If no density is given, molar volume/molecular weight is used from xml
    """

    def __init__(self,
                 exp_csv_filename,
                 param_xml_file,
                 x_guess,
                 h_guess,
                 phase_names,
                 aq_solvent_name,
                 extractant_name,
                 diluant_name,
                 aq_solvent_rho=None,
                 extractant_rho=None,
                 diluant_rho=None,
                 ):
        self._exp_csv_filename = exp_csv_filename
        self._x_guess = x_guess
        self._h_guess = h_guess
        self._aq_solvent_name = aq_solvent_name
        self._extractant_name = extractant_name
        self._diluant_name = diluant_name
        self._aq_solvent_rho = aq_solvent_rho
        self._extractant_rho = extractant_rho
        self._diluant_rho = diluant_rho

        self._phases = ct.import_phases(param_xml_file, phase_names)
        self._exp_df = pd.read_csv(self._exp_csv_filename)
        self._in_moles = None

        self.set_in_moles()

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

    def set_in_moles(self):
        phases_copy = self._phases.copy()
        exp_df = self._exp_df.copy()
        solvent_name = self._aq_solvent_name
        extractant_name = self._extractant_name
        diluant_name = self._diluant_name
        solvent_rho = self._aq_solvent_rho
        extractant_rho = self._extractant_rho
        diluant_rho = self._diluant_rho

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
        extractant_ind = phases_copy[org_ind].species_names.index(
            extractant_name)
        diluant_ind = phases_copy[org_ind].species_names.index(diluant_name)

        mix_aq = mixed.phase(aq_ind)
        mix_org = mixed.phase(org_ind)
        solvent_mw = mix_aq.molecular_weights[solvent_ind]  # g/mol
        extractant_mw = mix_org.molecular_weights[extractant_ind]
        diluant_mw = mix_org.molecular_weights[diluant_ind]
        if solvent_rho is None:
            solvent_rho = mix_aq(aq_ind).partial_molar_volumes[
                                            solvent_ind]/solvent_mw * 1e6  # g/L
            self._aq_solvent_rho = solvent_rho
        if extractant_rho is None:
            extractant_rho = mix_org(org_ind).partial_molar_volumes[
                                            extractant_ind]/extractant_mw * 1e6
            self._extractant_rho = extractant_rho
        if diluant_rho is None:
            diluant_rho = mix_org(org_ind).partial_molar_volumes[
                                 extractant_ind] / extractant_mw * 1e6
            self._diluant_rho = diluant_rho
            
        in_moles_data = []
        feed_vol = 1.  # g/L
        aq_phase_solvent_moles = feed_vol * solvent_rho / solvent_mw
        for row in exp_df.values:
            h_plus_moles = feed_vol * row[0]
            hydroxide_ions = 0
            rare_earth_moles = feed_vol * row[6]
            chlorine_moles = 3 * rare_earth_moles + h_plus_moles
            extractant_moles = feed_vol * row[3]
            extractant_vol = extractant_moles * extractant_rho / extractant_mw
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
