import os
import csv
from datetime import datetime
from typing import Optional


class ResultLogger:

    def __init__(self, name):
        self.base_dir = "../results"
        self.fields = ["Name", "Trial", "Round", "Context", "Action", "Reward", "Regret", "Time"]
        self.name = name
        self.log_dir: Optional[str] = None
        self.log_file_path: Optional[str] = None

        self.curr_sim_name = ""
        self.curr_sim_trial = 1

    def new_log(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_dir = os.path.join(self.base_dir, f"run_{self.name}_{timestamp}")
        os.makedirs(self.log_dir, exist_ok=False)

        self.log_file_path = os.path.join(self.log_dir, "log.csv")

        with open(self.log_file_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(self.fields)

    def log(self, t, context, action, reward, regret):

        row = [self.curr_sim_name, self.curr_sim_trial, t, context, action, reward, regret, datetime.now().strftime("%Y/%m/%d-%H:%M:%S")]

        if not self.log_file_path:
            raise RuntimeError("Log file not initialized. Call new_log() first.")

        with open(self.log_file_path, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(row)

    # def log(self, t, reward, regret):
    #     row = [self.curr_sim_name, self.curr_sim_trial, t, "", "", reward, regret,
    #            datetime.now().strftime("%Y/%m/%d-%H:%M:%S")]
    #
    #     if not self.log_file_path:
    #         raise RuntimeError("Log file not initialized. Call new_log() first.")
    #
    #     with open(self.log_file_path, mode="a", newline="", encoding="utf-8") as f:
    #         writer = csv.writer(f)
    #         writer.writerow(row)

    def set_simulation(self, name, trial):
        self.curr_sim_name = name
        self.curr_sim_trial = trial

    def get_results_dir(self, file_name : str = "") -> str:
        return os.path.join(self.log_dir, file_name)