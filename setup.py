#  LLEPE: Liquid-Liquid Equilibrium Parameter Estimator
#  Copyright (C) 2020, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See LICENSE for more details.

from setuptools import setup

setup(
    name='llepe',
    version='0.0.0',
    packages=['llepe'],
    package_data={
        'csvs': ['data/csvs/*.csv'],
        'xmls': ['data/xmls/*.xml'],
        'tests': ['tests/*.txt'],
    },
    include_package_data=True,
    zip_safe=False,
    url='',
    license='modified BSD license',
    author='UChicago Argonne, LLC.',
    author_email='',
    description='Liquid-liquid extraction parameter searcher',
    install_requires=['cantera==2.4.0',
                      'pandas==1.0.3',
                      'numpy==1.22.0',
                      'scipy==1.4.1',
                      'matplotlib==3.1.3']
)
