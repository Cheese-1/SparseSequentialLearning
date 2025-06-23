from utility.Logger import ResultLogger
from utility.SettingsSimulator import SettingsSimulator
from utility.Visualizer import Visualizer
import matplotlib
#matplotlib.use("tkagg")
import numpy as np
import matplotlib.pyplot as plt

def main():

    # simple_example()
    complex_simulation()

    # vz = Visualizer("../results/run_Subspace Learning_20250527_101414", True, False)
    # vz.generate_big_branin()
    # vz.generate_big_linear()
    # vz.generate_condition_linear(0,0)
    # vz.generate_condition_linear(0, 2)
    # vz.generate_condition_branin(0,0)
    # vz.generate_condition_branin(0, 2)
    # vz.generate_summary_linear(0)
    # vz.generate_summary_linear(1)
    # vz.generate_summary_branin(0)
    # vz.generate_summary_branin(1)


def complex_simulation():

    settings_dir = "../configurations"
    simulator = SettingsSimulator(settings_dir, "simulation_config.json")

    simulator.simulate_all()

if __name__ == "__main__":
    main()
