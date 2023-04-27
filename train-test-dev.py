from .IQL.runner import IQLRunner
from .AWAC.runner import AWACRunner
import os
import json

config_path = {'iql': 'config/iql.json',
               'awac': 'config/awac.json'}

with open(config_path['iql'], 'r') as file:
    config = json.load(file)
    IQLRunner(**config)

with open(config_path['awac'], 'r') as file:
    config = json.load(file)
    AWACRunner(**config)
