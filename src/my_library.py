import sys, os
import json
from pathlib import Path

#########################################################
#########################################################
# Create a json file based on config (python dict)
# and save it in a specified destination (absolute address)
def save_config_to_file(config, destination, indent=1):
    destination = Path(destination)
    # create containing folder is necessary
    if not destination.parent.exists():
        os.makedirs(destination.parent)
    # if destination file exists, delete it
    if destination.exists():
        os.remove(str(destination))
    # write config object
    with open(str(destination), 'w') as outfile:
        json.dump(config, outfile, indent=indent)

#########################################################
#########################################################
# Enter function description
def f():
    pass