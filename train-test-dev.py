from IQL.runner import IQLRunner
from AWAC.runner import AWACRunner
import os
import json

root = os.path.abspath('.') + '/RL/'
config_path = {'iql': root + 'configs/iql.json',
               'awac': root + 'configs/awac.json'}

with open(config_path['iql'], 'r') as file:
    config = json.load(file)
    IQLRunner(**config)

with open(config_path['awac'], 'r') as file:
    config = json.load(file)
    AWACRunner(**config)
