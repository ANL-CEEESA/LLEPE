from setuptools import setup

setup(
    name='reeps',
    version='0.0.0',
    packages=['reeps'],
    package_data={
        'csvs': ['data/csvs/*.csv'],
        'xmls': ['data/xmls/*.xml']
    },
    include_package_data=True,
    zip_safe=False,
    url='',
    license='',
    author='Titus Quah',
    author_email='',
    description='Rare earth element parameter searcher',
    install_requires=['cantera==2.4.0',
                      'pandas==1.0.3',
                      'numpy==1.15.4',
                      'scipy==1.4.1',
                      'seaborn==0.10.1',
                      'matplotlib==3.1.3']
)
