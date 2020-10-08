# LLEPE
LLEPE (Liquid-Liquid Extraction Parameter Estimator) is a toolkit for estimating standard thermodynamic parameters for Gibbs minimization.



## Installation

To install llepe, clone the repository with the command

```
$ git clone https://xgitlab.cels.anl.gov/summer-2020/parameter-estimation.git
```

Navigate into the parameter-estimation folder with 
```
cd parameter-estimation
```
and run the command below in your terminal
```
$ pip install -e.
```
For docs and tests, run
```
$ pip install -e .[docs,tests]
```
### Dependencies
llepe uses packages: cantera (https://cantera.org/), pandas, numpy, scipy, xml, seaborn, and matplotlib

## Usage
Check out examples in docs/examples

Readthedocs documentation are here: https://llepe.readthedocs.io/en/latest/index.html
```python
from llepe import LLEPE
opt_dict = {'Nd(H(A)2)3(org)_h0': {'upper_element_name': 'species',
						  'upper_attrib_name': 'name',
						  'upper_attrib_value': 'Nd(H(A)2)3(org)',
						  'lower_element_name': 'h0',
						  'lower_attrib_name': None,
						  'lower_attrib_value': None,
						  'input_format': '{0}',
						  'input_value': -4.7e6}}

searcher_parameters = {'exp_data': 'Nd_exp_data.csv',
					   'phases_xml_filename': 'twophase.xml',
					   'opt_dict': opt_dict,
					   'phase_names': ['HCl_electrolyte', 'PC88A_liquid'],
					   'aq_solvent_name': 'H2O(L)',
					   'extractant_name': '(HA)2(org)',
					   'diluant_name': 'dodecane',
					   'complex_names': ['Nd(H(A)2)3(org)'],
					   'extracted_species_ion_names': ['Nd+++'],
					   'aq_solvent_rho': 1000.0,
					   'extractant_rho': 960.0,
					   'diluant_rho': 750.0}
searcher = LLEPE(**searcher_parameters)
est_enthalpy = searcher.fit()
searcher.update_xml(est_enthalpy)
searcher.parity_plot(print_r_squared=True)
```
