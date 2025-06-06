from utility.Logger import ResultLogger
from utility.SettingsSimulator import SettingsSimulator
from utility.Visualizer import Visualizer
import matplotlib
matplotlib.use("tkagg")
import numpy as np
import matplotlib.pyplot as plt

def main():

    # simple_example()
    complex_simulation()

    # vz = Visualizer("../results/run_Subspace Learning_20250527_101414", True, True)
    # vz.generate_graphs()


def complex_simulation():

    settings_dir = "../configurations"
    simulator = SettingsSimulator(settings_dir, "SIBO_config.json")

    simulator.simulate_all()

if __name__ == "__main__":
    main()
