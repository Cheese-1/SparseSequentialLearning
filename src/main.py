from utility.SettingsSimulator import SettingsSimulator
from utility.Visualizer import Visualizer

def main():

    # Simulate the configuration provided in the simulation_config.json file.
    complex_simulation()

    # vz = Visualizer("../results/run_Subspace Learning_20250527_101414", True, False)
    # vz._generate_big_branin()
    # vz._generate_big_linear()
    # vz._generate_condition_linear(0,0)
    # vz._generate_condition_linear(0, 2)
    # vz._generate_condition_branin(0,0)
    # vz._generate_condition_branin(0, 2)
    # vz._generate_summary_linear(0)
    # vz._generate_summary_linear(1)
    # vz._generate_summary_branin(0)
    # vz._generate_summary_branin(1)


def complex_simulation():
    """
    A simulation of the bandit algorithm in an environment. The specification for the simulation
    are extracted from the configurations/simulation_config.json file.
    """
    settings_dir = "../configurations"
    simulator = SettingsSimulator(settings_dir, "simulation_config.json")

    simulator.simulate_all()

if __name__ == "__main__":
    main()
