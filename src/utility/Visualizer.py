import os

import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import numpy as np

class Visualizer:

    def __init__(self, result_directory, do_export = False, do_show = False):
        """
        Initialize the Visual handler.

        :param result_directory: the filepath to the generated result directory
        :param do_export: a flag to indicate whether to export the graphs in the result directory
        :param do_show: a flag to indicate whether to show the graphs after generation.
        """

        self.dir = result_directory
        self.log_filepath = os.path.join(self.dir, "log.csv")
        self.figures_dir = os.path.join(self.dir, "figures")

        os.makedirs(self.figures_dir, exist_ok=True)

        self.do_export = do_export
        self.do_show = do_show

        self.w = 11 / 2
        self.h = 8.5 / 2

    def generate_graphs(self):
        """
        Generates generic regret graphs for a given result file.

        """
        df = pd.read_csv(self.log_filepath, sep=',', header=0, encoding='utf-8')

        # Project to reduce compute resources
        df = df[['Name', 'Trial', 'Round', 'Reward', 'Regret']]

        # Sort the dataframe for convenience
        df.sort_values(['Name', 'Trial', 'Round'])

        # Compute the cumulative reward
        df['cum_reward'] = df.groupby(['Name', 'Trial'])['Reward'].cumsum()
        df['cum_regret'] = df.groupby(['Name', 'Trial'])['Regret'].cumsum()
        df['cum_min_regret'] = df.groupby(['Name', 'Trial'])['Regret'].cummin()
        df['cum_avg_regret'] = df['cum_regret']  / df['Round']

        data = (
            df
            .groupby(['Name', 'Round'])[['cum_reward', 'cum_regret', 'cum_min_regret', 'cum_avg_regret']]
            .agg(
                avg_cum_reward=('cum_reward', 'mean'),
                std_cum_reward=('cum_reward', 'std'),
                avg_cum_regret=('cum_regret', 'mean'),
                std_cum_regret=('cum_regret', 'std'),
                avg_cum_min_regret=('cum_min_regret', 'mean'),
                std_cum_min_regret=('cum_min_regret', 'std'),
                avg_cum_avg_regret=('cum_avg_regret', 'mean'),
                std_cum_avg_regret=('cum_avg_regret', 'std'),
            )
            .reset_index()
        )

        # Generate The graphs
        self._generate_reward_graph(data)
        self._generate_regret_graphs(data)

    def _generate_reward_graph(self, data):

        names = data['Name'].to_list()

        plt.figure()

        for name in set(names):

            time = data.loc[data['Name'] == name, 'Round'].to_numpy()
            reward = data.loc[data['Name'] == name,'avg_cum_reward'].to_numpy()
            std_reward = data.loc[data['Name'] == name,'std_cum_reward'].to_numpy()

            plt.plot(time, reward, label=f"{name}")
            plt.fill_between(time, reward - std_reward, reward + std_reward, alpha=0.3)

        plt.xlabel('Round t')
        plt.ylabel('Cumulative Reward')
        plt.title("Cumulative Reward across Rounds")
        plt.legend()
        plt.tight_layout()

        self.do_export and plt.savefig(os.path.join(self.figures_dir, "cumulative_reward.png"), dpi=300, bbox_inches='tight', format='png')
        self.do_show and (plt.show())

    def _generate_regret_graphs(self, data):

        names = data['Name'].to_numpy()

        plt.figure()

        for name in set(names):

            time = data.loc[data['Name'] == name,'Round'].to_numpy()
            cum_regret = data.loc[data['Name'] == name,'avg_cum_regret'].to_numpy()
            std_cum_regret = data.loc[data['Name'] == name,'std_cum_regret'].to_numpy()

            plt.plot(time, cum_regret, label=f"{name}")
            plt.fill_between(time, cum_regret - std_cum_regret, cum_regret + std_cum_regret, alpha=0.3)


        plt.xlabel('Round t')
        plt.ylabel('Cumulative Regret')
        plt.title("Cumulative Regret across Rounds")
        plt.legend()
        plt.tight_layout()

        self.do_export and plt.savefig(os.path.join(self.figures_dir, "cumulative_regret.png"), dpi=300, bbox_inches='tight', format='png')
        self.do_show and plt.show()

        plt.figure()

        for name in set(names):

            time = data.loc[data['Name'] == name, 'Round'].to_numpy()
            simp_regret = data.loc[data['Name'] == name,'avg_cum_min_regret'].to_numpy()
            std_simp_regret = data.loc[data['Name'] == name,'std_cum_min_regret'].to_numpy()

            plt.plot(time, simp_regret, label=f"{name}")
            plt.fill_between(time, simp_regret - std_simp_regret, simp_regret + std_simp_regret, alpha=0.3)

        plt.xlabel('Round t')
        plt.ylabel('Simple Regret')
        plt.title("Simple Regret across Rounds")
        plt.legend()
        plt.tight_layout()

        self.do_export and plt.savefig(os.path.join(self.figures_dir, "simple_regret.png"), dpi=300, bbox_inches='tight', format='png')
        self.do_show and plt.show()

        plt.figure()

        for name in set(names):

            time = data.loc[data['Name'] == name, 'Round'].to_numpy()
            avg_regret = data.loc[data['Name'] == name,'avg_cum_avg_regret'].to_numpy()
            std_avg_regret = data.loc[data['Name'] == name,'std_cum_avg_regret'].to_numpy()

            plt.plot(time, avg_regret, label=f"{name}")
            plt.fill_between(time, avg_regret - std_avg_regret, avg_regret + std_avg_regret, alpha=0.3)

        plt.xlabel('Round t')
        plt.ylabel('Average Regret')
        plt.title("Average Regret across Rounds")
        plt.legend()
        plt.tight_layout()

        self.do_export and plt.savefig(os.path.join(self.figures_dir, "average_regret.png"), dpi=300, bbox_inches='tight', format='png')
        self.do_show and plt.show()

    def _generate_big_branin(self):
        repos = ["run_Subspace Learning_20250614_085325", "run_Subspace Learning_20250610_080749"]

        df1 = pd.read_csv(os.path.join("../results/Branin", repos[0], "log.csv"), sep=',', header=0, encoding='utf-8')
        df2 = pd.read_csv(os.path.join("../results/Branin", repos[1], "log.csv"), sep=',', header=0, encoding='utf-8')

        df = pd.concat([df1, df2], ignore_index=True)

        df = df[['Name', 'Trial', 'Round', 'Reward', 'Regret']]

        # Sort the dataframe for convenience
        df.sort_values(['Name', 'Trial', 'Round'])

        df['cum_regret'] = df.groupby(['Name', 'Trial'])['Regret'].cumsum()
        df['cum_avg_regret'] = df['cum_regret'] / df['Round']

        data = (
            df
            .groupby(['Name', 'Round'])[['cum_avg_regret']]
            .agg(
                avg_cum_avg_regret=('cum_avg_regret', 'mean'),
                std_cum_avg_regret=('cum_avg_regret', 'std'),
            )
            .reset_index()
        )

        names = set(data['Name'].to_list())

        distributions = [
            ["SIBKB-RBF,d=5", 'SIBO-RBF,d=5'], ["SIBKB-RBF,d=10", 'SIBO-RBF,d=10'], ["SIBKB-RBF,d=25", 'SIBO-RBF,d=25'],
            ["SIBKB-Matern,d=5", 'SIBO-Matern,d=5'], ["SIBKB-Matern,d=10", 'SIBO-Matern,d=10'],
            ["SIBKB-Matern,d=25", 'SIBO-Matern,d=25'],
            ["SIBKB-Quad,d=5", 'SIBO-Quad,d=5'], ["SIBKB-Quad,d=10", 'SIBO-Quad,d=10'],
            ["SIBKB-Quad,d=25", 'SIBO-Quad,d=25']
        ]

        mpl.rcParams['font.family'] = 'serif'
        mpl.rcParams['font.serif'] = ['Times New Roman',
                                      'Palatino Linotype',
                                      'Computer Modern Roman']

        fig, axes = plt.subplots(
            nrows=3, ncols=3,
            figsize=(11, 11)
        )

        row_labels = ['RBF', 'Matérn', 'RQ']
        col_labels = ['60%', '80%', '92%']

        for i in range(3):  # row index
            for j in range(3):  # column index
                ax = axes[i, j]

                lbls = ["SI-BKB", "SI-BO"]
                clrs = ["blue", 'red']
                lnstls = ["--", "-"]

                c = 0
                for name in distributions[3 * j + i]:
                    time = data.loc[data['Name'] == name, 'Round'].to_numpy()
                    avg_regret = data.loc[data['Name'] == name, 'avg_cum_avg_regret'].to_numpy()
                    std_avg_regret = data.loc[data['Name'] == name, 'std_cum_avg_regret'].to_numpy()

                    time = time[:2000]
                    avg_regret = avg_regret[:2000]
                    std_avg_regret = std_avg_regret[:2000]

                    ax.plot(time, avg_regret, label=f"{lbls[c]}", linestyle=lnstls[c], color=clrs[c], zorder=(2 - c))
                    ax.set_ylim((0, 125))
                    ax.set_yticks([0, 25, 50, 75, 100, 125])
                    ax.fill_between(time, avg_regret - std_avg_regret, avg_regret + std_avg_regret, alpha=0.3)
                    c += 1

                # per-panel axis titles
                ax.legend(fontsize='small')
                ax.set_xlabel(r'Round $t$', fontsize=9)
                ax.set_ylabel(r'Regret $R_t/t$', fontsize=9)

        # Add the big graph's column ticks
        for ax, txt in zip(axes[-1], col_labels):
            # two-line label: regular x-label + ambient-dimension tick
            ax.annotate(txt, xy=(0.5, -0.35), xycoords='axes fraction',
                        ha='center', va='bottom', fontsize=13)

        # Add the big graph's row ticks
        for ax, txt in zip(axes[:, 0], row_labels):
            # annotate just outside the plotting area so the y-label can stay
            ax.annotate(txt, xy=(-0.25, 0.5), xycoords='axes fraction',
                        ha='right', va='center', fontsize=13, rotation=90)

        # Add title and axis titles to the big graph
        big_ax = fig.add_subplot(111, frameon=False)
        big_ax.tick_params(labelcolor='none', top=False, bottom=False,
                           left=False, right=False)
        big_ax.set_xlabel(r'Sparsity $(d - k)/d$ (%)', fontsize=15, labelpad=40)
        big_ax.set_ylabel(r'Tuned Kernel $\kappa$', fontsize=15, labelpad=45)

        big_ax.set_title(
            'The effect of the kernel choice and ambient dimension \n on the regret '
            'performance in a Branin Bandit.',
            fontsize=17,
            y=1.03
        )

        fig.tight_layout(rect=(0.02, 0.02, 0.98, 0.98))

        self.do_export and plt.savefig(os.path.join("..", "generated_graphs", "branin", f"LargeBraninBandit.pdf"),
                                       bbox_inches='tight', format='pdf')
        self.do_show and plt.show()

        # Clear the figure
        plt.figure().clear()


    def _generate_big_linear(self):

        repos = ["run_Subspace Learning_20250525_190309", "run_Subspace Learning_20250609_180356", "run_Subspace Learning_20250609_204328"]

        df1 = pd.read_csv(os.path.join("../results/Linear", repos[0], "log.csv"), sep=',', header=0, encoding='utf-8')
        df2 = pd.read_csv(os.path.join("../results/Linear", repos[1], "log.csv"), sep=',', header=0, encoding='utf-8')
        df3 = pd.read_csv(os.path.join("../results/Linear", repos[2], "log.csv"), sep=',', header=0, encoding='utf-8')

        df = pd.concat([df1, df2, df3], ignore_index=True)

        df = df[['Name', 'Trial', 'Round', 'Reward', 'Regret']]

        # Sort the dataframe for convenience
        df.sort_values(['Name', 'Trial', 'Round'])

        df['cum_regret'] = df.groupby(['Name', 'Trial'])['Regret'].cumsum()
        df['cum_avg_regret'] = df['cum_regret'] / df['Round']

        data = (
            df
            .groupby(['Name', 'Round'])[['cum_avg_regret']]
            .agg(
                avg_cum_avg_regret=('cum_avg_regret', 'mean'),
                std_cum_avg_regret=('cum_avg_regret', 'std'),
            )
            .reset_index()
        )
        names = set(data['Name'].to_list())

        distributions = [
            ['SIBKB-RBF-d=5','SIBO-RBF-d=5'], ['SIBKB-RBF-d=10','SIBO-RBF-d=10'], ['SIBKB-RBF-d=25','SIBO - RBF'],
            ['SIBKB-Matern-d=5','SIBO-Matern-d=5'], ['SIBKB-Matern-d=10', 'SIBO-Matern-d=10'], ['SIBKB-Matern-d=25', 'SIBO - Matern'],
            ['SIBKB-Quad-d=5', 'SIBO-Quad-d=5'], ['SIBKB-Quad-d=10', 'SIBO-Quad-d=10'], ['SIBKB-Quad-d=25','SIBO - Rational Quadratic' ]
         ]

        mpl.rcParams['font.family'] = 'serif'
        mpl.rcParams['font.serif'] = ['Times New Roman',
                                      'Palatino Linotype',
                                      'Computer Modern Roman']



        fig, axes = plt.subplots(
            nrows=3, ncols=3,
            figsize=(11, 11)  # square figure, comfortable for papers
        )

        row_labels = ['RBF', 'Matérn', 'RQ']
        col_labels = ['40%', '70%', '88%']

        for i in range(3):  # row index
            for j in range(3):  # column index
                ax = axes[i, j]

                lbls = ["SI-BKB", "SI-BO"]
                clrs = ["blue", 'red']
                lnstls = ["--", "-"]


                c = 0
                for name in distributions[3 * j + i]:
                    time = data.loc[data['Name'] == name, 'Round'].to_numpy()
                    avg_regret = data.loc[data['Name'] == name, 'avg_cum_avg_regret'].to_numpy()
                    std_avg_regret = data.loc[data['Name'] == name, 'std_cum_avg_regret'].to_numpy()

                    time = time[:2000]
                    avg_regret = avg_regret[:2000]
                    std_avg_regret = std_avg_regret[:2000]

                    ax.plot(time, avg_regret, label=f"{lbls[c]}", linestyle = lnstls[c], color=clrs[c], zorder=(2 - c))
                    ax.set_ylim((0,550))
                    ax.set_yticks([0,100,200,300,400,500])
                    ax.fill_between(time, avg_regret - std_avg_regret, avg_regret + std_avg_regret, alpha=0.3)
                    c += 1

                # per-panel axis titles
                ax.legend(fontsize='small')
                ax.set_xlabel(r'Round $t$', fontsize=9)
                ax.set_ylabel(r'Regret $R_t/t$', fontsize=9)

        # Add the big graph's column ticks
        for ax, txt in zip(axes[-1], col_labels):
            # two-line label: regular x-label + ambient-dimension tick
            ax.annotate(txt, xy=(0.5, -0.35), xycoords='axes fraction',
                        ha='center', va='bottom', fontsize=13)

        # Add the big graph's row ticks
        for ax, txt in zip(axes[:, 0], row_labels):
            # annotate just outside the plotting area so the y-label can stay
            ax.annotate(txt, xy=(-0.25, 0.5), xycoords='axes fraction',
                        ha='right', va='center', fontsize=13, rotation=90)

        # Add title and axis titles to the big graph
        big_ax = fig.add_subplot(111, frameon=False)
        big_ax.tick_params(labelcolor='none', top=False, bottom=False,
                           left=False, right=False)
        big_ax.set_xlabel(r'Sparsity $(d-k)/d$ (%)', fontsize=15, labelpad=40)
        big_ax.set_ylabel(r'Tuned Kernel $\kappa$', fontsize=15, labelpad=45)

        big_ax.set_title(
            'The effect of the kernel choice and ambient dimension \n on the regret '
            'performance in a Linear Bandit.',
            fontsize=17,
            y=1.03
        )

        fig.tight_layout(rect=(0.02, 0.02, 0.98, 0.98))

        self.do_export and plt.savefig(os.path.join("..", "generated_graphs", "linear", f"LargeLinBandit.pdf"),
                                       bbox_inches='tight', format='pdf')
        self.do_show and plt.show()

        # Clear the figure
        plt.figure().clear()


    def _generate_condition_linear(self, i, j):
        repos = ["run_Subspace Learning_20250525_190309", "run_Subspace Learning_20250609_180356",
                 "run_Subspace Learning_20250609_204328"]

        df1 = pd.read_csv(os.path.join("../results/Linear", repos[0], "log.csv"), sep=',', header=0, encoding='utf-8')
        df2 = pd.read_csv(os.path.join("../results/Linear", repos[1], "log.csv"), sep=',', header=0, encoding='utf-8')
        df3 = pd.read_csv(os.path.join("../results/Linear", repos[2], "log.csv"), sep=',', header=0, encoding='utf-8')

        df = pd.concat([df1, df2, df3], ignore_index=True)

        df = df[['Name', 'Trial', 'Round', 'Reward', 'Regret']]

        # Sort the dataframe for convenience
        df.sort_values(['Name', 'Trial', 'Round'])

        df['cum_regret'] = df.groupby(['Name', 'Trial'])['Regret'].cumsum()
        df['cum_avg_regret'] = df['cum_regret'] / df['Round']

        data = (
            df
            .groupby(['Name', 'Round'])[['cum_avg_regret']]
            .agg(
                avg_cum_avg_regret=('cum_avg_regret', 'mean'),
                std_cum_avg_regret=('cum_avg_regret', 'std'),
            )
            .reset_index()
        )

        names = set(data['Name'].to_list())

        distributions = [
            ['SIBKB-RBF-d=5', 'SIBO-RBF-d=5'], ['SIBKB-RBF-d=10', 'SIBO-RBF-d=10'], ['SIBKB-RBF-d=25', 'SIBO - RBF'],
            ['SIBKB-Matern-d=5', 'SIBO-Matern-d=5'], ['SIBKB-Matern-d=10', 'SIBO-Matern-d=10'],
            ['SIBKB-Matern-d=25', 'SIBO - Matern'],
            ['SIBKB-Quad-d=5', 'SIBO-Quad-d=5'], ['SIBKB-Quad-d=10', 'SIBO-Quad-d=10'],
            ['SIBKB-Quad-d=25', 'SIBO - Rational Quadratic']
        ]

        mpl.rcParams['font.family'] = 'serif'
        mpl.rcParams['font.serif'] = ['Times New Roman',
                                      'Palatino Linotype',
                                      'Computer Modern Roman']


        row_labels = ['RBF', 'Matérn', 'RQ']
        col_labels = ['40%', '70%', '88%']


        lbls = ["SI-BKB", "SI-BO"]
        clrs = ["blue", 'red']
        lnstls = ["--", "-"]

        plt.figure(figsize=(self.w, self.h))
        c = 0
        for name in distributions[3 * j + i]:
            time = data.loc[data['Name'] == name, 'Round'].to_numpy()
            avg_regret = data.loc[data['Name'] == name, 'avg_cum_avg_regret'].to_numpy()
            std_avg_regret = data.loc[data['Name'] == name, 'std_cum_avg_regret'].to_numpy()

            time = time[:2000]
            avg_regret = avg_regret[:2000]
            std_avg_regret = std_avg_regret[:2000]

            plt.plot(time, avg_regret, label=f"{lbls[c]}", linestyle=lnstls[c], color=clrs[c], zorder=(2 - c))
            plt.ylim((0, 550))
            plt.yticks([0, 100, 200, 300, 400, 500])
            plt.fill_between(time, avg_regret - 1 * std_avg_regret, avg_regret + 1 * std_avg_regret, alpha=0.3)
            c += 1

        # Add titles, legend, and grid
        plt.legend(fontsize='small')
        plt.xlabel(r'Round $t$', fontsize=11)
        plt.ylabel(r'Regret $R_t/t$', fontsize=11)

        plt.grid()
        plt.title(
            f"{row_labels[i]} Kernel at {col_labels[j]} Sparsity",
            fontsize=11,
            y = 1.02
        )

        self.do_export and plt.savefig(os.path.join("..", "generated_graphs", "linear", f"Lin{i}{j}Bandit.pdf"),
                                       bbox_inches='tight', format='pdf')
        self.do_show and plt.show()

        # Clear the figure
        plt.figure().clear()


    def _generate_condition_branin(self, i, j):
        repos = ["run_Subspace Learning_20250614_085325", "run_Subspace Learning_20250610_080749"]

        df1 = pd.read_csv(os.path.join("../results/Branin", repos[0], "log.csv"), sep=',', header=0, encoding='utf-8')
        df2 = pd.read_csv(os.path.join("../results/Branin", repos[1], "log.csv"), sep=',', header=0, encoding='utf-8')

        df = pd.concat([df1, df2], ignore_index=True)

        df = df[['Name', 'Trial', 'Round', 'Reward', 'Regret']]

        # Sort the dataframe for convenience
        df.sort_values(['Name', 'Trial', 'Round'])

        df['cum_regret'] = df.groupby(['Name', 'Trial'])['Regret'].cumsum()
        df['cum_avg_regret'] = df['cum_regret'] / df['Round']

        data = (
            df
            .groupby(['Name', 'Round'])[['cum_avg_regret']]
            .agg(
                avg_cum_avg_regret=('cum_avg_regret', 'mean'),
                std_cum_avg_regret=('cum_avg_regret', 'std'),
            )
            .reset_index()
        )

        names = set(data['Name'].to_list())

        distributions = [
            ["SIBKB-RBF,d=5", 'SIBO-RBF,d=5'], ["SIBKB-RBF,d=10", 'SIBO-RBF,d=10'], ["SIBKB-RBF,d=25", 'SIBO-RBF,d=25'],
            ["SIBKB-Matern,d=5", 'SIBO-Matern,d=5'], ["SIBKB-Matern,d=10", 'SIBO-Matern,d=10'],
            ["SIBKB-Matern,d=25", 'SIBO-Matern,d=25'],
            ["SIBKB-Quad,d=5", 'SIBO-Quad,d=5'], ["SIBKB-Quad,d=10", 'SIBO-Quad,d=10'],
            ["SIBKB-Quad,d=25", 'SIBO-Quad,d=25']
        ]
        mpl.rcParams['font.family'] = 'serif'
        mpl.rcParams['font.serif'] = ['Times New Roman',
                                      'Palatino Linotype',
                                      'Computer Modern Roman']


        row_labels = ['RBF', 'Matérn', 'RQ']
        col_labels = ['60%', '80%', '92%']

        plt.figure(figsize=(self.w, self.h))

        lbls = ["SI-BKB", "SI-BO"]
        clrs = ["blue", 'red']
        lnstls = ["--", "-"]

        c = 0
        for name in distributions[3 * j + i]:
            time = data.loc[data['Name'] == name, 'Round'].to_numpy()
            avg_regret = data.loc[data['Name'] == name, 'avg_cum_avg_regret'].to_numpy()
            std_avg_regret = data.loc[data['Name'] == name, 'std_cum_avg_regret'].to_numpy()

            time = time[:2000]
            avg_regret = avg_regret[:2000]
            std_avg_regret = std_avg_regret[:2000]

            plt.plot(time, avg_regret, label=f"{lbls[c]}", linestyle=lnstls[c], color=clrs[c], zorder=(2 - c))
            plt.ylim((0, 125))
            plt.yticks([0, 25, 50, 75, 100, 125])
            plt.fill_between(time, avg_regret - 1 * std_avg_regret, avg_regret + 1 * std_avg_regret, alpha=0.3)
            c += 1

        # Add titles, legend, and grid
        plt.legend(fontsize='small')
        plt.xlabel(r'Round $t$', fontsize=11)
        plt.ylabel(r'Regret $R_t/t$', fontsize=11)

        plt.grid()
        plt.title(
            f"{row_labels[i]} Kernel at {col_labels[j]} Sparsity",
            fontsize=11,
            y = 1.02
        )

        self.do_export and plt.savefig(os.path.join("..", "generated_graphs", "branin", f"Branin{i}{j}Bandit.pdf"),
                                       bbox_inches='tight', format='pdf')
        self.do_show and plt.show()

        # Clear the figure
        plt.figure().clear()

    def _generate_summary_linear(self, m):
        repos = ["run_Subspace Learning_20250525_190309", "run_Subspace Learning_20250609_180356",
                 "run_Subspace Learning_20250609_204328"]

        df1 = pd.read_csv(os.path.join("../results/Linear", repos[0], "log.csv"), sep=',', header=0, encoding='utf-8')
        df2 = pd.read_csv(os.path.join("../results/Linear", repos[1], "log.csv"), sep=',', header=0, encoding='utf-8')
        df3 = pd.read_csv(os.path.join("../results/Linear", repos[2], "log.csv"), sep=',', header=0, encoding='utf-8')

        df = pd.concat([df1, df2, df3], ignore_index=True)

        df = df[['Name', 'Trial', 'Round', 'Reward', 'Regret']]

        # Sort the dataframe for convenience
        df.sort_values(['Name', 'Trial', 'Round'])

        df['cum_regret'] = df.groupby(['Name', 'Trial'])['Regret'].cumsum()
        df['cum_avg_regret'] = df['cum_regret'] / df['Round']

        data = (
            df
            .groupby(['Name', 'Round'])[['cum_avg_regret']]
            .agg(
                avg_cum_avg_regret=('cum_avg_regret', 'mean'),
                std_cum_avg_regret=('cum_avg_regret', 'std'),
            )
            .reset_index()
        )

        names = set(data['Name'].to_list())

        distributions = [
            ['SIBKB-RBF-d=5', 'SIBO-RBF-d=5'], ['SIBKB-RBF-d=10', 'SIBO-RBF-d=10'], ['SIBKB-RBF-d=25', 'SIBO - RBF'],
            ['SIBKB-Matern-d=5', 'SIBO-Matern-d=5'], ['SIBKB-Matern-d=10', 'SIBO-Matern-d=10'],
            ['SIBKB-Matern-d=25', 'SIBO - Matern'],
            ['SIBKB-Quad-d=5', 'SIBO-Quad-d=5'], ['SIBKB-Quad-d=10', 'SIBO-Quad-d=10'],
            ['SIBKB-Quad-d=25', 'SIBO - Rational Quadratic']
        ]

        mpl.rcParams['font.family'] = 'serif'
        mpl.rcParams['font.serif'] = ['Times New Roman',
                                      'Palatino Linotype',
                                      'Computer Modern Roman']


        plt.figure(figsize=(self.w, self.h))

        row_labels = ['RBF Kernel', 'Matérn Kernel', 'RQ Kernel']
        col_values = [100 - 60, 100 - 30, 100 - 12]
        c = 0
        for i in range(3):  # row index

            ys = []
            ys_std = []

            lbls = ["SI-BKB", "SI-BO"]
            clrs = ["blue", 'red', 'green']
            lnstls = ["-o", "--^", ":s"]

            for j in range(3):  # column index

                name = distributions[3 * j + i][m]
                avg_regret = data.loc[data['Name'] == name, 'avg_cum_avg_regret'].to_numpy()
                std_avg_regret = data.loc[data['Name'] == name, 'std_cum_avg_regret'].to_numpy()

                avg_regret = np.average(avg_regret[1901:2000])

                # Compute the error on the average of averages
                std_avg_regret = 0.1 * np.sqrt(np.average(std_avg_regret[1901:2000]** 2))

                ys.append(avg_regret)
                ys_std.append(std_avg_regret)

            plt.errorbar(col_values, ys,yerr = ys_std, label=f"{row_labels[i]}", color=clrs[i], fmt=lnstls[i], capsize=5)

        # Add titles, legend and grid
        plt.legend(fontsize='small')
        plt.xlabel(r'Sparsity $(d - k)/d$ (%)', fontsize=9)
        plt.ylabel(r'Average Cumulative Regret $\overline{R_T/T}$', fontsize=9)
        plt.grid()

        plt.title(
            f"{lbls[m]} Algorithm",
            fontsize=11,
            y = 1.02
        )

        self.do_export and plt.savefig(os.path.join("..", "generated_graphs", "linear", f"Summary{m}LinBandit.pdf"),
                                       format='pdf')
        self.do_show and plt.show()

        # Clear the figure
        plt.figure().clear()

    def _generate_summary_branin(self, m):
        repos = ["run_Subspace Learning_20250614_085325", "run_Subspace Learning_20250610_080749"]

        df1 = pd.read_csv(os.path.join("../results/Branin", repos[0], "log.csv"), sep=',', header=0, encoding='utf-8')
        df2 = pd.read_csv(os.path.join("../results/Branin", repos[1], "log.csv"), sep=',', header=0, encoding='utf-8')

        df = pd.concat([df1, df2], ignore_index=True)

        df = df[['Name', 'Trial', 'Round', 'Reward', 'Regret']]

        # Sort the dataframe for convenience
        df.sort_values(['Name', 'Trial', 'Round'])

        df['cum_regret'] = df.groupby(['Name', 'Trial'])['Regret'].cumsum()
        df['cum_avg_regret'] = df['cum_regret'] / df['Round']

        data = (
            df
            .groupby(['Name', 'Round'])[['cum_avg_regret']]
            .agg(
                avg_cum_avg_regret=('cum_avg_regret', 'mean'),
                std_cum_avg_regret=('cum_avg_regret', 'std'),
            )
            .reset_index()
        )

        names = set(data['Name'].to_list())

        distributions = [
            ["SIBKB-RBF,d=5", 'SIBO-RBF,d=5'], ["SIBKB-RBF,d=10", 'SIBO-RBF,d=10'], ["SIBKB-RBF,d=25", 'SIBO-RBF,d=25'],
            ["SIBKB-Matern,d=5", 'SIBO-Matern,d=5'], ["SIBKB-Matern,d=10", 'SIBO-Matern,d=10'],
            ["SIBKB-Matern,d=25", 'SIBO-Matern,d=25'],
            ["SIBKB-Quad,d=5", 'SIBO-Quad,d=5'], ["SIBKB-Quad,d=10", 'SIBO-Quad,d=10'],
            ["SIBKB-Quad,d=25", 'SIBO-Quad,d=25']
        ]
        mpl.rcParams['font.family'] = 'serif'
        mpl.rcParams['font.serif'] = ['Times New Roman',
                                      'Palatino Linotype',
                                      'Computer Modern Roman']

        plt.figure(figsize=(self.w, self.h))

        row_labels = ['RBF', 'Matérn', 'RQ']
        col_values = [100 - 40, 100 - 20, 100 - 8]
        c = 0

        for i in range(3):  # row index

            ys = []
            ys_std = []

            lbls = ["SI-BKB", "SI-BO"]
            clrs = ["blue", 'red', 'green']
            lnstls = ["-o", "--^", ":s"]

            for j in range(3):  # column index

                name = distributions[3 * j + i][m]
                avg_regret = data.loc[data['Name'] == name, 'avg_cum_avg_regret'].to_numpy()
                std_avg_regret = data.loc[data['Name'] == name, 'std_cum_avg_regret'].to_numpy()

                avg_regret = np.average(avg_regret[1901:2000])

                # Compute the error on the average of averages
                std_avg_regret = 0.1 * np.sqrt(np.average(std_avg_regret[1901:2000]** 2))

                ys.append(avg_regret)
                ys_std.append(std_avg_regret)

            plt.errorbar(col_values, ys,yerr = ys_std, label=f"{row_labels[i]}", color=clrs[i], fmt=lnstls[i], capsize=5)

        # Add titles, grid, and legend
        plt.legend(fontsize='small')
        plt.xlabel(r'Sparsity $(d - k)/d$ (%)', fontsize=9)
        plt.ylabel(r'Average Cumulative Regret $\overline{R_T/T}$', fontsize=9)
        plt.grid()

        plt.title(
            f"{lbls[m]} Algorithm",
            fontsize=11,
            y = 1.02
        )

        self.do_export and plt.savefig(os.path.join("..", "generated_graphs", "branin", f"Summary{m}BraninBandit.pdf"),
                                       format='pdf')
        self.do_show and plt.show()

        # Clear the figure
        plt.figure().clear()
