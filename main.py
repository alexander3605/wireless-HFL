import sys, os
sys.path.insert(1, os.path.join(os.getcwd(), "src"))
from os import listdir
from os.path import isfile, join
from simulator import Simulator
import warnings
warnings.filterwarnings('ignore')


CONFIG_FILES_DIR = join(os.getcwd(), "config_files")


def run_simulator(config_folder):
    config_files = [join(config_folder,f) for f in listdir(config_folder) if isfile(join(config_folder, f))]
    # print(config_files)
    print("====", "Executing simulations in folder:", config_folder, sep="\t")
    print("====", "Configuration files found:\t", len(config_files), sep="\t")
    print()
    simulator = Simulator(config_files)
    simulator.start()


if __name__ == "__main__":
    run_simulator(CONFIG_FILES_DIR)