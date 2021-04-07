.. LLEPE: Liquid-Liquid Equilibrium Parameter Estimator
   Copyright (C) 2020, UChicago Argonne, LLC. All rights reserved.
   Released under the modified BSD license. See LICENSE for more details.

Welcome to LLEPE's docs! - the Liquid-Liquid Extraction Parameter Estimator
===========================================================================
LLEPE is a package for thermodynamic parameter estimation for liquid-liquid extraction modeling

LLEPE takes experimental data in a csv and system data in a xml.

The package then uses Cantera, another python package, to simulate equilibrium.

Error between predicted and experimental data is then minimized.

.. toctree::
   :maxdepth: 2
   :caption: User Guide
   
   guide/install
   guide/quickstart
   guide/about
   
.. toctree::
   :maxdepth: 1
   :caption: Estimators
   
   modules/LLEPE
   




Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
