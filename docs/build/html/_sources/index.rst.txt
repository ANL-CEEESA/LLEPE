.. reeps documentation master file, created by Titus Quah
   sphinx-quickstart on Tue Jun  9 10:13:23 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to reeps's docs! - the Rare earth element parameter searcher
====================================================================
REEPS is a package for thermodynamic parameter estimation specifically 
for rare earth element extraction modeling. 

REEPS takes experimental data in a csv and system data in a xml.

The package then uses Cantera, another python package, to simulate equilibrium.

Error between predicted and experimental data is then minimized.

.. toctree::
   :maxdepth: 2
   :caption: User Guide
   
   guide/install
   guide/quickstart
   
.. toctree::
   :maxdepth: 1
   :caption: Searchers
   
   modules/reeps
   




Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
