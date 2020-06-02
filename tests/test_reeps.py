import sys

sys.path.append('../')
from reeps import REEPS
import json

with open('one_comp_settings.txt') as file:
    testing_params = json.load(file)
beaker = REEPS(**testing_params)
print(beaker.get_in_moles()['Nd+++'])