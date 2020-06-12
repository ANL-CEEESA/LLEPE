.. _quickstart:

***************
Getting Started
***************

Here is a quick example of how to fit an xml thermodynamic model to experimental data.

This code fits Nd standard enthalpy in the "twophase.xml" cantera file to the 
experimental data in "Nd_exp_data.csv".

The code then produces a parity plot of the measured and predicted concentrations of Nd 3+ in the 
aqueous phase.

.. code-block:: python

	from reeps import REEPS

	searcher_parameters = {'exp_csv_filename': 'Nd_exp_data.csv',
						   'phases_xml_filename': 'twophase.xml',
						   'opt_dict': {'Nd(H(A)2)3(org)': {'h0': -4.7e6}},
						   'phase_names': ['HCl_electrolyte', 'PC88A_liquid'],
						   'aq_solvent_name': 'H2O(L)',
						   'extractant_name': '(HA)2(org)',
						   'diluant_name': 'dodecane',
						   'complex_names': ['Nd(H(A)2)3(org)'],
						   'rare_earth_ion_names': ['Nd+++'],
						   'aq_solvent_rho': 1000.0,
						   'extractant_rho': 960.0,
						   'diluant_rho': 750.0}
	searcher = REEPS(**searcher_parameters)
	est_enthalpy = searcher.fit()
	searcher.update_xml(est_enthalpy)
	searcher.parity_plot(print_r_squared=True)
	
The code should return something like this
