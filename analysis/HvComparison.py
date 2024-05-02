import os
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import config
import glob
import config

from pymoo.indicators.hv import HV
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting


analysis_root = os.path.join(config.root_dir, 'analysis')
outdir = os.path.join(analysis_root, 'output')

class HvComparison:

    def __init__(self, analysis_dir='unconstrained_5x5'):
        self.nds = NonDominatedSorting()
        self.ref_point = np.array(config.hv_ref_point)  # science, cost
        self.hv_client = HV(self.ref_point)

        self.analysis_dir = os.path.join(analysis_root, analysis_dir)
        self.ga_path = os.path.join(self.analysis_dir, 'ga')
        self.rl_paths = [
            # os.path.join(config.root_dir, 'model', 'rl', 'ppo', 'runs_mo'),
            # os.path.join(config.root_dir, 'model', 'rl', 'ppo', 'runs_mo_2b'),
            # os.path.join(config.root_dir, 'model', 'rl', 'ppo', 'runs_mo_6b'),
            # os.path.join(config.root_dir, 'model', 'rl', 'ppo', 'runs_mo_12b'),
            # os.path.join(config.root_dir, 'model', 'rl', 'ppo', 'runs_store', 'mo_20b_2w'),
            os.path.join(self.analysis_dir, 'ppo_36task'),
        ]
        self.rl_path_transfer = os.path.join(self.analysis_dir, 'ppo_36task')
        self.rl_path = os.path.join(self.analysis_dir, 'ppo')


    def run_statistics(self):
        rl_hv_data = HvComparison.get_hv_data(self.rl_path_transfer)
        rl_min_nfe, rl_max_nfe = HvComparison.get_nfe_data2(self.rl_path_transfer)
        print('Pretrain RL HV Mean:', np.mean(rl_hv_data))
        print('Pretrain RL HV Std:', np.std(rl_hv_data))
        print('Pretrain RL Avg Max NFE:', np.mean(rl_max_nfe))

        rl2_hv_data = HvComparison.get_hv_data(self.rl_path)
        rl2_min_nfe, rl2_max_nfe = HvComparison.get_nfe_data2(self.rl_path)
        print('RL HV Mean:', np.mean(rl2_hv_data))
        print('RL HV Std:', np.std(rl2_hv_data))
        print('RL Avg Max NFE:', np.mean(rl2_max_nfe))


        # print(ga_hv_data)
        ga_hv_data = HvComparison.get_hv_data(self.ga_path)
        print('GA HV Mean:', np.mean(ga_hv_data))
        print('GA HV Std:', np.std(ga_hv_data))





    def run_line_comparison(self, title='Line Comparison', ga_label='NSGA-II'):
        traces = []
        traces.extend(HvComparison.get_dataframe_interp(self.ga_path, ga_label, unique_nfe=False))
        df = pd.concat(traces, ignore_index=True)
        df[['nfe', 'hv']] = df[['nfe', 'hv']].apply(pd.to_numeric)
        HvComparison.plot_hv_line_comparison(df, self.rl_paths[0], plot_name='hv_comparison_lines.png', title=title)



    def run_ci_comparison(self, title='Confidence Interval Comparison', rl_label='PPO', ga_label='NSGA-II'):
        traces = []
        for p in self.rl_paths:
            traces.extend(HvComparison.get_dataframe_interp(p, rl_label, unique_nfe=False))
        traces.extend(HvComparison.get_dataframe_interp(self.ga_path, ga_label, unique_nfe=False))

        # Combine dataframes
        df = pd.concat(traces, ignore_index=True)
        df[['nfe', 'hv']] = df[['nfe', 'hv']].apply(pd.to_numeric)

        HvComparison.plot_hv_comparison(
            df,
            plot_name='hv_comparison.png',
            title=title
        )

    def run_3ci_comparison(self, title='Confidence Interval Comparison', rl_label='PPO', ga_label='NSGA-II', rl_label2='PPO2'):
        traces = []
        traces.extend(HvComparison.get_dataframe_interp(self.rl_path_transfer, rl_label, unique_nfe=False))
        traces.extend(HvComparison.get_dataframe_interp(self.rl_path, rl_label2, unique_nfe=False))
        traces.extend(HvComparison.get_dataframe_interp(self.ga_path, ga_label, unique_nfe=False))

        # Combine dataframes
        df = pd.concat(traces, ignore_index=True)
        df[['nfe', 'hv']] = df[['nfe', 'hv']].apply(pd.to_numeric)

        HvComparison.plot_hv_comparison(
            df,
            plot_name='hv_comparison.png',
            title=title
        )

    # --------------------------------------
    # Plotting
    # --------------------------------------

    def plot_rl_pareto(self, rl_dir):
        rl_run_name = os.path.basename(rl_dir)
        gens_files = HvComparison.get_pop_files(rl_dir)
        all_designs = []
        all_designs_norm = []
        all_designs_bitstr = []
        for gens_file in gens_files:
            with open(gens_file, 'r') as file:
                data = json.load(file)
                for design in data:
                    bitstr = design['design']
                    science_norm = design['science']
                    cost_norm = design['cost']
                    science = science_norm * 0.425 * -1.0
                    cost = cost_norm * 2.5e4
                    all_designs.append([science, cost])
                    all_designs_norm.append([science_norm, cost_norm])
                    all_designs_bitstr.append(bitstr)

        F = np.array(all_designs_norm)
        fronts = self.nds.do(F, n_stop_if_ranked=30)
        survivors = []
        for k, front in enumerate(fronts, start=1):
            for idx in front:
                survivors.append(idx)
            break  # only get first front

        combined_science = []
        combined_cost = []
        combined_bitstr = []
        for idx in survivors:
            combined_science.append(all_designs[idx][0])
            combined_cost.append(all_designs[idx][1])
            combined_bitstr.append(all_designs_bitstr[idx])

        plt.cla()
        plt.scatter(combined_science, combined_cost, s=5, label=rl_run_name + ' Pareto')
        plt.title(rl_run_name + ' Combined Pareto: hv = ' + str(self.hv_client.do(F)))
        plt.xlabel('Science')
        plt.ylabel('Cost')
        plt.xlim([-0.1, 0.7])  # Replace x_min and x_max with your desired values
        plt.ylim([0, 30000])
        plt.legend()
        save_path = os.path.join(outdir, os.path.basename(rl_dir) + '_combined_pareto.png')
        plt.savefig(save_path)

        combined_designs = []
        for bits, science, cost in zip(combined_bitstr, combined_science, combined_cost):
            combined_designs.append({'bitstr': bits, 'science': science, 'cost': cost})
        # write to json
        with open(os.path.join(outdir, os.path.basename(rl_dir) + '_combined_pareto.json'), 'w') as file:
            json.dump(combined_designs, file, indent=4)

    def plot_ga_pareto(self):
        gens_files = HvComparison.get_gens_files(self.ga_path)
        all_designs = []
        all_designs_norm = []
        all_designs_bitstr = []
        for gens_file in gens_files:
            with open(gens_file, 'r') as file:
                data = json.load(file)
                gens = data['generations']
                last_gen = gens[-1]
                lg_designs = last_gen['designs']
                for lg_design in lg_designs:
                    if lg_design['design'] not in all_designs_bitstr:
                        all_designs.append(lg_design['objectives'])
                        all_designs_norm.append(lg_design['objectives_norm'])
                        all_designs_bitstr.append(lg_design['design'])

        F = np.array(all_designs_norm)
        fronts = self.nds.do(F, n_stop_if_ranked=30)
        survivors = []
        for k, front in enumerate(fronts, start=1):
            for idx in front:
                survivors.append(idx)
            break  # only get first front

        combined_science = []
        combined_cost = []
        combined_bitstr = []
        for idx in survivors:
            combined_science.append(all_designs[idx][0])
            combined_cost.append(all_designs[idx][1])
            combined_bitstr.append(all_designs_bitstr[idx])

        plt.cla()
        plt.scatter(combined_science, combined_cost, s=5, label='GA Pareto')
        plt.title('GA Combined Pareto: hv = ' + str(self.hv_client.do(F)))
        plt.xlabel('Science')
        plt.ylabel('Cost')
        plt.xlim([-0.1, 0.7])  # Replace x_min and x_max with your desired values
        plt.ylim([0, 30000])
        plt.legend()
        save_path = os.path.join(outdir, 'ga_combined_pareto.png')
        plt.savefig(save_path)

        combined_designs = []
        for bits, science, cost in zip(combined_bitstr, combined_science, combined_cost):
            combined_designs.append({'bitstr': bits, 'science': science, 'cost': cost})
        # write to json
        with open(os.path.join(outdir, 'ga_combined_pareto.json'), 'w') as file:
            json.dump(combined_designs, file, indent=4)

    # --------------------------------------
    # Static Plotting
    # --------------------------------------

    @staticmethod
    def plot_hv_comparison(data_frames, plot_name='hv_comparison.png', title='Hypervolume Comparison'):
        # Plotting
        plt.clf()
        sns.set(style="darkgrid")
        plt.figure(figsize=(8, 5))
        # sns.lineplot(x='nfe', y='hv', hue='label', data=data_frames, ci='sd', estimator='mean')
        sns.lineplot(x='nfe', y='hv', hue='label', data=data_frames, ci='sd', estimator='mean', linewidth=2.5)
        plt.title(title, fontsize=20)
        plt.xlabel('NFE', fontsize=16)
        plt.ylabel('Hypervolume', fontsize=16)
        plt.xticks(fontsize=14)  # Larger x-axis tick labels
        plt.yticks(fontsize=14)  # Larger y-axis tick labels
        plt.legend(fontsize=14, loc='lower right')
        save_path = os.path.join(outdir, plot_name)
        plt.savefig(save_path)

    @staticmethod
    def plot_hv_line_comparison(data_frames, lines_path, plot_name='hv_comparison_lines.png', title='Line Comparison'):

        # Get lines hv files
        hv_files = HvComparison.get_hv_files(lines_path)


        # Plotting
        plt.clf()
        sns.set(style="darkgrid")
        plt.figure(figsize=(8, 5))
        sns.lineplot(x='nfe', y='hv', hue='label', data=data_frames, ci='sd', estimator='mean', linewidth=2.5)
        for hv_file in hv_files:
            with open(hv_file, 'r') as file:
                data = json.load(file)
                nfe = [item[0] for item in data]
                hv = [item[1] for item in data]
                # plt.plot(nfe, hv, label=os.path.basename(hv_file))

                # plot line but make very transparent
                plt.plot(nfe, hv, label=os.path.basename(hv_file), alpha=1.0, linewidth=1.5)

        plt.title(title, fontsize=20)
        plt.xlabel('NFE', fontsize=16)
        plt.ylabel('Hypervolume', fontsize=16)
        plt.xticks(fontsize=14)  # Larger x-axis tick labels
        plt.yticks(fontsize=14)  # Larger y-axis tick labels
        # plt.legend(fontsize=14, loc='lower right')
        save_path = os.path.join(outdir, plot_name)
        plt.savefig(save_path)

    @staticmethod
    def plot_run_hvs(base_dir, plot_name='hv_runs.png'):
        plt.clf()
        hv_files = HvComparison.get_hv_files(base_dir)
        for idx, hv_file in enumerate(hv_files):
            with open(hv_file, 'r') as file:
                data = json.load(file)
                nfe = [item[0] for item in data]
                hv = [item[1] for item in data]
                plt.plot(nfe, hv, label=str(idx))

        plt.xlabel('NFE')
        plt.ylabel('HV')
        plt.legend()

        save_path = os.path.join(outdir, plot_name)
        plt.savefig(save_path)

    @staticmethod
    def plot_combined_pop(base_dir, plot_name='run_pops.png'):
        plt.clf()
        pop_files = HvComparison.get_pop_files(base_dir)
        all_science = []
        all_cost = []
        for idx, pop_file in enumerate(pop_files):
            with open(pop_file, 'r') as file:
                data = json.load(file)
                science = [item['science'] * 0.425 for item in data]
                cost = [item['cost'] * 2.5e4 for item in data]
                all_science.extend(science)
                all_cost.extend(cost)
                plt.scatter(science, cost, s=5, label=os.path.basename(pop_file))
        # plt.scatter(all_science, all_cost, s=5)
        plt.xlabel('Science')
        plt.ylabel('Cost')
        plt.xlim([-0.1, 0.7])  # Replace x_min and x_max with your desired values
        plt.ylim([0, 30000])
        plt.legend()
        save_path = os.path.join(outdir, plot_name)
        plt.savefig(save_path)

    # --------------------------------------
    # Static Helpers
    # --------------------------------------

    @staticmethod
    def get_gens_files(base_dir):
        pattern = os.path.join(base_dir, '**', 'gens*.json')
        gens_files = glob.glob(pattern, recursive=True)
        return gens_files

    @staticmethod
    def get_hv_files(base_dir):
        pattern = os.path.join(base_dir, '**', 'hv*.json')
        hv_files = glob.glob(pattern, recursive=True)
        return hv_files

    @staticmethod
    def get_pop_files(base_dir):
        pattern = os.path.join(base_dir, '**', 'pop*.json')
        pop_files = glob.glob(pattern, recursive=True)
        return pop_files

    @staticmethod
    def get_nfe_data(base_dir):
        min_nfe = []
        max_nfe = []
        hv_files = HvComparison.get_hv_files(base_dir)
        for hv_file in hv_files:
            with open(hv_file, 'r') as file:
                data = json.load(file)
                nfe = [item[0] for item in data]
                hv = [item[1] for item in data]
                min_nfe.append(min(nfe))
                max_nfe.append(max(nfe))
        min_nfe = min(min_nfe)
        max_nfe = max(max_nfe)
        steps = max_nfe - min_nfe
        return min_nfe, max_nfe, steps

    @staticmethod
    def get_nfe_data2(base_dir):
        min_nfe = []
        max_nfe = []
        hv_files = HvComparison.get_hv_files(base_dir)
        for hv_file in hv_files:
            with open(hv_file, 'r') as file:
                data = json.load(file)
                nfe = [item[0] for item in data]
                hv = [item[1] for item in data]
                min_nfe.append(min(nfe))
                max_nfe.append(max(nfe))
        return min_nfe, max_nfe

    @staticmethod
    def get_hv_data(base_dir):
        max_hv = []
        hv_files = HvComparison.get_hv_files(base_dir)
        for hv_file in hv_files:
            with open(hv_file, 'r') as file:
                data = json.load(file)
                nfe = [item[0] for item in data]
                hv = [item[1] for item in data]
                max_hv.append(max(hv))
        return max_hv



    @staticmethod
    def get_dataframe_interp(base_dir, label, unique_nfe=False):

        # Get interpolation data
        min_nfe, max_nfe, steps = HvComparison.get_nfe_data(base_dir)

        print(base_dir, min_nfe, max_nfe, steps)

        # Linear interpolation for smooth plotting
        hv_files = HvComparison.get_hv_files(base_dir)
        dfs = []
        for hv_file in hv_files:
            if os.path.exists(hv_file):
                with open(hv_file, 'r') as file:
                    data = json.load(file)
                    nfe = [item[0] for item in data]
                    hv = [item[1] for item in data]

                    if unique_nfe is True:
                        # Unique nfes
                        d_min_nfe = min(nfe)
                        d_max_nfe = max(nfe)
                        d_steps = d_max_nfe - d_min_nfe
                        nfe_space = np.linspace(d_min_nfe, d_max_nfe, d_steps)
                    else:
                        # Aggregate nfes
                        nfe_space = np.linspace(min_nfe, max_nfe, steps)


                    hv_interp = np.interp(nfe_space, nfe, hv)
                    run_df = pd.DataFrame({'nfe': nfe_space, 'hv': hv_interp, 'label': label})
                    dfs.append(run_df)
        return dfs

    @staticmethod
    def get_dataframe(base_dir, label):
        hv_files = HvComparison.get_hv_files(base_dir)
        dfs = []
        for hv_file in hv_files:
            if os.path.exists(hv_file):
                with open(hv_file, 'r') as file:
                    data = json.load(file)
                    nfe = [item[0] for item in data]
                    hv = [item[1] for item in data]
                    run_df = pd.DataFrame({'nfe': nfe, 'hv': hv, 'label': label})
                    dfs.append(run_df)
        return dfs



def rename_files(directory_path):
    # Check if the provided path is a directory
    if not os.path.isdir(directory_path):
        print("The provided path is not a directory.")
        return

    for filename in os.listdir(directory_path):
        # Check if the file is a JSON file
        if filename.endswith('.json'):
            # Constructing the new file name
            new_filename = filename.replace('uniform', 'hv_uniform')

            # Construct the full path for the original and new file names
            original_file = os.path.join(directory_path, filename)
            new_file = os.path.join(directory_path, new_filename)

            # Renaming the file
            os.rename(original_file, new_file)
            print(f"Renamed {filename} to {new_filename}")


if __name__ == '__main__':
    # rename_path = os.path.join(config.root_dir, 'results', 'save', 'ga_results')
    # rename_files(rename_path)

    parse_dir = 'unconstrained_5x5'


    hv_comparison = HvComparison(parse_dir)

    rl_label = 'Generative Agent (pretrained)'
    ga_label = 'NSGA-II'


    ci_title = '5x5 Truss Validation: Generative Agent vs NSGA-II'
    hv_comparison.run_ci_comparison(title=ci_title, rl_label=rl_label, ga_label=ga_label)
    # hv_comparison.run_3ci_comparison(title=ci_title, rl_label=rl_label, ga_label=ga_label, rl_label2='Generative Agent (random init)')

    line_title = '5x5 Truss Validation: Generative Agent vs NSGA-II'
    hv_comparison.run_line_comparison(title=line_title, ga_label=ga_label)


    hv_comparison.run_statistics()




















