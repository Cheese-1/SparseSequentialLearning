from src.utility.Logger import ResultLogger
from src.utility.Visualizer import Visualizer
from src import Environments, Learners
from src.Environments import AbstractEnvironment
from src.Learners import AbstractLearner

import os.path
import json
from datetime import datetime
from tqdm import trange



class SettingsSimulator:

    def __init__(self, settings_dir, filename="simulation_config.json"):

        self.settings_dir = settings_dir
        self.filename = filename
        self.settings_path = os.path.join(settings_dir, self.filename)

        self._read_settings()

        self.logger = ResultLogger(self.name)
        self.logger.new_log()

        self._replicate_settings(self.logger.get_results_dir(self.filename))
        self.visualizer : Visualizer = Visualizer(self.logger.log_dir, self.do_export, self.do_show)


    def _read_settings(self):

        data = json.load(open(self.settings_path, mode = "r", encoding="utf-8"))

        assert data["simulations"] is not None
        assert data["name"] is not None

        self.name = data["name"]
        self.do_export = data["export_figures"]
        self.do_show = data["show_figures"]
        self.settings = data["simulations"]

        self.curr_simulation = 0
        self.num_simulations = len(self.settings)

        simulation_names = list(map(lambda sim : sim["name"], self.settings))

        # Make sure that simulation names are unique
        if len(simulation_names) != len(set(simulation_names)):
            raise RuntimeError("Simulation names are not unique")

        self.run_names = simulation_names

    def _replicate_settings(self, file_path : str):

        # Determine the total number of trials
        total_trials = sum(map(lambda x : x["trials"], self.settings))

        data = {
            "name" : self.name,
            "date" : datetime.now().strftime("%Y/%m/%d-%H:%M:%S"),

            "number of simulations" : self.num_simulations,
            "number of trials" : total_trials,
            "simulation names" : self.run_names,

            "simulations" : self.settings
        }

        with open(file_path, mode="w", encoding="utf-8") as f:
            json.dump(data, f, indent = 4)


    def simulate_next(self):

        if not self._has_next_simulation():
            return

        curr_settings = self.settings[self.curr_simulation]

        # Extract the parameters
        name = curr_settings["name"]
        trials = curr_settings["trials"]
        n = curr_settings["horizon"]

        env_cls = getattr(Environments, curr_settings["env"])
        learner_cls = getattr(Learners, curr_settings["learner"])


        for trial in trange(trials, desc=f"Running {name}"):
            # Set up the logger
            self.logger.set_simulation(name, trial + 1)

            # Instantiate a new copy of the environment and learner
            env : AbstractEnvironment = env_cls(curr_settings["env_config"])
            learner : AbstractLearner = learner_cls(n, curr_settings["learner_config"])

            learner.run(env, self.logger)

            # Remove the copy
            del env
            del learner

        self.curr_simulation += 1

    def simulate_all(self):

        while self._has_next_simulation():
            self.simulate_next()

        self.visualizer.generate_graphs()

    def _has_next_simulation(self):
        return self.curr_simulation < self.num_simulations