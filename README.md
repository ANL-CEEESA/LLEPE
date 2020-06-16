# REEPS
REEPS (Rare Earth Element Parameter Searcher) is a toolkit for estimating standard thermodynamic parameters for Gibbs minimization.
Extend a methodology for estimating standard thermodynamic parameters for Gibbs minimization in multiphase, multicomponent separations systems


## Installation

To install REEPS, clone the repository with the command

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
### Dependencies
REEPS uses packages: cantera (https://cantera.org/), pandas, numpy, scipy, xml, seaborn, and matplotlib

## Usage
Check out examples in docs/examples
```python
from reeps import REEPS
searcher = REEPS(**REEPS_parameters_dictionary)
optimized_parameter_dictionary = searcher.fit()
searcher.update_xml(optimized_parameter_dictionary)
searcher.parity_plot()
print(seacher.r_squared())
```