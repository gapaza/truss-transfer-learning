import os
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import config
import glob
import config
from pandas.plotting import table

from pymoo.indicators.hv import HV
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting


studies_dir = os.path.join(config.root_dir, 'taskC', 'studies')
analysis_root = os.path.join(studies_dir, 'analysis')
outdir = os.path.join(analysis_root, 'output')

class AnalysisClient:

    def __init__(self, result_dir='constrained_3x3'):
        self.nds = NonDominatedSorting()
        self.ref_point = np.array(config.hv_ref_point)  # science, cost
        self.hv_client = HV(self.ref_point)

        self.analysis_dir = result_dir
        self.ga_path = os.path.join(self.analysis_dir, 'ga')
        self.ppo_base_path = os.path.join(self.analysis_dir, 'ppo-base')
        self.ppo_transfer_path = os.path.join(self.analysis_dir, 'ppo-transfer')

        # self.rl_paths = [self.ppo_base_path, self.ppo_transfer_path]
        # self.rl_path = self.ppo_base_path
        # self.rl_path_transfer = self.ppo_transfer_path


    def load_dir_results(self, dir_path):
        hv_files = AnalysisClient.get_hv_files(dir_path)
        all_hvs = []
        for hv_file in hv_files:
            with open(hv_file, 'r') as file:
                data = json.load(file)
                nfe = [item[0] for item in data]
                hv = [item[1] for item in data]
                all_hvs.append(hv)
        return all_hvs



    def run_statistics(self, read_path):
        data = AnalysisClient.get_hv_data(read_path)
        print('HV Mean:', np.mean(data))
        print('HV Std:', np.std(data))
        mean = round(np.mean(data), 3)
        std = round(np.std(data), 3)
        return mean, std




    # def run_line_comparison(self, title='Line Comparison', ga_label='NSGA-II'):
    #     traces = []
    #     traces.extend(HvComparison.get_dataframe_interp(self.ga_path, ga_label, unique_nfe=False))
    #     df = pd.concat(traces, ignore_index=True)
    #     df[['nfe', 'hv']] = df[['nfe', 'hv']].apply(pd.to_numeric)
    #     HvComparison.plot_hv_line_comparison(df, self.rl_paths[0], plot_name='hv_comparison_lines.png', title=title)



    def run_ci_comparison(self, title='Confidence Interval Comparison', rl_label='PPO', ga_label='NSGA-II'):
        if len(os.listdir(self.ppo_transfer_path)) == 0:
            return
        if len(os.listdir(self.ga_path)) == 0:
            return

        traces = []
        all_paths = [self.ppo_transfer_path]
        for p in all_paths:
            traces.extend(AnalysisClient.get_dataframe_interp(p, rl_label, unique_nfe=False))
        traces.extend(AnalysisClient.get_dataframe_interp(self.ga_path, ga_label, unique_nfe=False))

        # Combine dataframes
        df = pd.concat(traces, ignore_index=True)
        df[['nfe', 'hv']] = df[['nfe', 'hv']].apply(pd.to_numeric)

        # Get table statistics
        ppoT_avg, ppoT_std = self.run_statistics(self.ppo_transfer_path)  # Integer values
        ga_avg, ga_std = self.run_statistics(self.ga_path)  # Integer values

        # Create pandas dataframe for table
        table_data = {
            'Algorithm': ['PPO Transfer', 'NSGA-II'],
            'Mean HV': [ppoT_avg, ga_avg],
            'Std HV': [ppoT_std, ga_std]
        }
        df_table = pd.DataFrame(table_data)

        AnalysisClient.plot_hv_comparison(
            df,
            # plot_name='ga-ppoT.png',
            title=title,
            plot_path=os.path.join(self.analysis_dir, 'ga-ppoT.png'),
            table_data=df_table
        )

    def run_3ci_comparison(self, title='Confidence Interval Comparison', rl_label='PPO', ga_label='NSGA-II', rl_label2='PPO2'):
        if len(os.listdir(self.ppo_transfer_path)) == 0:
            return
        if len(os.listdir(self.ga_path)) == 0:
            return
        if len(os.listdir(self.ppo_base_path)) == 0:
            return


        traces = []
        traces.extend(AnalysisClient.get_dataframe_interp(self.ppo_transfer_path, rl_label, unique_nfe=False))
        traces.extend(AnalysisClient.get_dataframe_interp(self.ga_path, ga_label, unique_nfe=False))
        traces.extend(AnalysisClient.get_dataframe_interp(self.ppo_base_path, rl_label2, unique_nfe=False))

        # Combine dataframes
        df = pd.concat(traces, ignore_index=True)
        df[['nfe', 'hv']] = df[['nfe', 'hv']].apply(pd.to_numeric)

        AnalysisClient.plot_hv_comparison(
            df,
            # plot_name='hv_comparison.png',
            title=title,
            plot_path=os.path.join(self.analysis_dir, 'ga-ppoT-ppoB.png')
        )

    # --------------------------------------
    # Plotting
    # --------------------------------------

    def plot_rl_pareto(self, rl_dir):
        rl_run_name = os.path.basename(rl_dir)
        gens_files = AnalysisClient.get_pop_files(rl_dir)
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
        gens_files = AnalysisClient.get_gens_files(self.ga_path)
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
    def plot_hv_comparison(data_frames, plot_name='hv_comparison.png', title='Hypervolume Comparison', plot_path=None, table_data=None):
        # Plotting


        plt.clf()
        sns.set(style="darkgrid")

        if table_data is not None:
            plt.figure(figsize=(8, 7))
            ax1 = plt.subplot2grid((8, 1), (0, 0), rowspan=6)  # Give more room for the plot
            sns.lineplot(x='nfe', y='hv', hue='label', data=data_frames, ci='sd', estimator='mean', linewidth=2.5, ax=ax1)
        else:
            plt.figure(figsize=(8, 5))
            sns.lineplot(x='nfe', y='hv', hue='label', data=data_frames, ci='sd', estimator='mean')

        plt.title(title, fontsize=20)
        plt.xlabel('NFE', fontsize=16)
        plt.ylabel('Hypervolume', fontsize=16)
        plt.xticks(fontsize=14)  # Larger x-axis tick labels
        plt.yticks(fontsize=14)  # Larger y-axis tick labels
        plt.legend(fontsize=14, loc='lower right')

        # Plotting the table below the line plot
        if table_data is not None:
            ax2 = plt.subplot2grid((8, 1), (6, 0), rowspan=2)  # More space for the table
            ax2.xaxis.set_visible(False)
            ax2.yaxis.set_visible(False)
            ax2.set_facecolor('none')  # Set face color to 'none' for the same color as the rest of the plot
            table = plt.table(cellText=table_data.values,
                              colLabels=table_data.columns,
                              loc='center',
                              cellLoc='center',
                              colColours=["palegreen"] * 3)  # You can customize colors
            table.auto_set_font_size(False)
            table.set_fontsize(12)
            table.scale(1, 1.75)  # You can adjust scaling to fit your layout
            plt.tight_layout()





        if plot_path is not None:
            save_path = plot_path
        else:
            save_path = os.path.join(outdir, plot_name)
        plt.savefig(save_path)

    @staticmethod
    def plot_hv_line_comparison(data_frames, lines_path, plot_name='hv_comparison_lines.png', title='Line Comparison'):

        # Get lines hv files
        hv_files = AnalysisClient.get_hv_files(lines_path)


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
        hv_files = AnalysisClient.get_hv_files(base_dir)
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
        pop_files = AnalysisClient.get_pop_files(base_dir)
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
        pattern = os.path.join(base_dir, '**', 'population*.json')
        pop_files = glob.glob(pattern, recursive=True)
        return pop_files

    @staticmethod
    def get_nfe_data(base_dir):
        min_nfe = []
        max_nfe = []
        print('base dir:', base_dir)
        hv_files = AnalysisClient.get_hv_files(base_dir)
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
        hv_files = AnalysisClient.get_hv_files(base_dir)
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
        hv_files = AnalysisClient.get_hv_files(base_dir)
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
        min_nfe, max_nfe, steps = AnalysisClient.get_nfe_data(base_dir)

        print(base_dir, min_nfe, max_nfe, steps)

        # Linear interpolation for smooth plotting
        hv_files = AnalysisClient.get_hv_files(base_dir)
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
        hv_files = AnalysisClient.get_hv_files(base_dir)
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

    study_num = 2    # 1, 2, 3
    problem_num = 2  # 0, 1, 2

    study = 'val' + str(study_num)
    problem = 'p' + str(problem_num)
    parse_dir = os.path.join(studies_dir, study, 'results', problem)

    client = AnalysisClient(parse_dir)

    ppo_transfer_label = 'Generative Agent (pretrained)'
    ppo_base_label = 'Generative Agent (random init)'
    ga_label = 'NSGA-II'

    ##########################################
    # Confidence Interval Comparison
    ##########################################

    # GA vs PPO Transfer
    ci_title = f"Validation {study_num}.{problem_num+1}"
    client.run_ci_comparison(title=ci_title, rl_label=ppo_transfer_label, ga_label=ga_label)

    # GA vs PPO Transfer vs PPO Base
    ci3_title = f"Validation {study_num}.{problem_num+1}"
    client.run_3ci_comparison(title=ci_title, rl_label=ppo_transfer_label, ga_label=ga_label, rl_label2=ppo_base_label)

    # line_title = '3x3 Truss Validation: Generative Agent vs NSGA-II'
    # hv_comparison.run_line_comparison(title=line_title, ga_label=ga_label)


    # hv_comparison.run_statistics()




















