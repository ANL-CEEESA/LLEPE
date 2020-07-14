from setuptools import setup

setup(
    name='llepe',
    version='0.0.0',
    packages=['LLEPE'],
    package_data={
        'csvs': ['data/csvs/*.csv'],
        'xmls': ['data/xmls/*.xml'],
        'tests': ['tests/*.txt'],
    },
    include_package_data=True,
    zip_safe=False,
    url='',
    license='',
    author='Titus Quah',
    author_email='',
    description='Liquid-liquid extraction parameter searcher',
    install_requires=['cantera==2.4.0',
                      'pandas==1.0.3',
                      'numpy==1.15.4',
                      'scipy==1.4.1',
                      'matplotlib==3.1.3']
)
