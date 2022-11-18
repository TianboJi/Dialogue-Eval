import argparse
import pandas as pd
import numpy as np
import json
from tqdm.auto import tqdm
from pathlib import Path
import random
from scipy.stats import (
    wilcoxon, ranksums, mannwhitneyu,
    pearsonr, spearmanr, kendalltau,
)
import scipy.stats as stats
from itertools import combinations, permutations, product
import matplotlib as mpl
import matplotlib.pyplot as plt
from collections import OrderedDict
import seaborn as sns
from tqdm.auto import tqdm


class MTurkStatistics:
    def __init__(self, dirname):
        dirpath = Path(dirname).absolute().resolve()
        self._preprocess(dirpath)

    def _preprocess(self, dirpath):
        print(f'preprocessing directory {dirpath}')
        self.mask_threshold = 0.1
        self.center_value = 0.05
        self.dirname = f"{dirpath}"
        self.dirpath = dirpath
        self.savedir = dirpath.joinpath("statistics")
        f_model_metadata = dirpath.joinpath('model_metadata.json')
        with f_model_metadata.open() as f:
            model_metadata = json.load(f)
        self.model_metadata = model_metadata
        self.modelnames = model_metadata['ranked_model']
        self.qc_model = model_metadata['qc_model']
        self.model_mappings = MTurkStatistics.model_mapping(self.modelnames)

        f_csv_passed = dirpath.joinpath('system_scores_passed.csv')
        self.pd_passed = pd.read_csv(f_csv_passed)
        f_csv_failed = dirpath.joinpath('system_scores_failed.csv')
        self.pd_failed = pd.read_csv(f_csv_failed)

        f_system_scores = dirpath.joinpath('system_scores.csv')
        self.pd_system_scores = pd.read_csv(f_system_scores)
        f_system_raw_scores = dirpath.joinpath('system_raw_scores.csv')
        self.pd_system_raw_scores = pd.read_csv(f_system_raw_scores)

    def process(self):

        self.compute_sigtest()
        self.compute_rater_agreement()
        self.get_corr_between_scores()

    def compute_sigtest(self):
        pd_passed = self.pd_passed
        modelnames = self.modelnames
        all_sig = []
        for m1 in modelnames:
            cur_sig = []
            for m2 in modelnames:
                if m1 == m2:
                    p = 1.0
                    cur_sig.append(p)
                    continue
                m1_pd = pd_passed[pd_passed['model'] == m1]
                m1_z = m1_pd.z.values

                m2_pd = pd_passed[pd_passed['model'] == m2]
                m2_z = m2_pd.z.values

                first = m1_z
                second = m2_z
                p = MTurkStatistics.sig_test(first, second)
                cur_sig.append(p)
            all_sig.append(cur_sig)
        sig_np = np.array(all_sig)
        self.sigtest_result = sig_np

    def draw_sigtest_figure(self):
        sigtest_result = self.sigtest_result
        mask_threshold = self.mask_threshold
        center_value = self.center_value
        ticklabels = [self.model_mappings[e] for e in self.modelnames]
        plt.figure()
        sns.set()
        sig_cmap = sns.color_palette("blend:#00A100,#cccc00", as_cmap=True)

        masked = sigtest_result > mask_threshold
        ax = sns.heatmap(
            sigtest_result,
            vmin=0.0,
            vmax=mask_threshold,
            cmap=sig_cmap,
            center=center_value,
            robust=True,
            linewidths=1,
            linecolor='white',
            mask=masked,
            cbar_kws={"ticks": [0, center_value, mask_threshold], "label": "", "shrink": 0.95},
            cbar=True,
            square=True,
        )
        ax.set_yticklabels(
            labels=ticklabels,
            ha='left',
            va='center',
            position=(-0.02, 0),
            rotation=None,
        )
        ax.set_xticklabels(
            labels=ticklabels,
            ha='center',
            va='top',
            position=(0, 0.02),
            rotation=None,
        )

        f_sig_test = self.savedir.joinpath('sig_test.pdf')
        plt.savefig(f_sig_test, bbox_inches='tight')
        plt.close()

    def compute_rater_agreement(self):
        pd_passed = self.pd_passed
        pd_failed = self.pd_failed

        passed_corr_dist = self.get_rater_agreement_distribution(pd_passed)
        failed_corr_dist = self.get_rater_agreement_distribution(pd_failed)

        pd_1 = pd.DataFrame()
        pd_1['correlation'] = passed_corr_dist
        pd_1['pass'] = 'passed'

        pd_2 = pd.DataFrame()
        pd_2['correlation'] = failed_corr_dist
        pd_2['pass'] = 'failed'

        pd_rater_agreement = pd.concat([pd_1, pd_2]).reset_index(drop=True)
        self.pd_rater_agreement = pd_rater_agreement

    def draw_rater_agreement_figure(self):
        pd_rater_agreement = self.pd_rater_agreement

        ax = sns.displot(
            data=pd_rater_agreement,
            x="correlation",
            kde=True,
            hue="pass",
            legend=False
        )
        # plt.ylim((0,120))
        plt.xlim((-1.0, 1.0))
        xts = [-1.0, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        plt.xticks(xts)
        plt.xlabel('$r$', fontsize=14)
        plt.ylabel(r'$\#$ workers', fontsize=14)

        f_rater_agreement = self.savedir.joinpath('rater_agreement.pdf')
        plt.savefig(f_rater_agreement, bbox_inches='tight')
        plt.close()

        ax = sns.displot(
            data=pd_rater_agreement,
            x="correlation",
            kde=True,
            hue="pass",
            legend=True
        )
        # plt.ylim((0,120))
        plt.xlim((-1.0, 1.0))
        xts = [-1.0, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        plt.xticks(xts)
        plt.xlabel('$r$', fontsize=14)
        plt.ylabel(r'$\#$ workers', fontsize=14)
        f_rater_agreement_with_legend = self.savedir.joinpath('rater_agreement_with_legend.pdf')
        plt.savefig(f_rater_agreement_with_legend, bbox_inches='tight')
        plt.close()

    def get_rater_agreement_distribution(self, pd_data):
        scores_model_combination = {}

        assignIDs = np.unique(pd_data.assignID.values)
        for assignID in assignIDs:
            pd_cur_assign = pd_data[pd_data['assignID'] == assignID].sort_values(by=['model'])
            zscores = pd_cur_assign['z'].values
            cur_model_combination = tuple(pd_cur_assign['model'].values.tolist())
            if cur_model_combination not in scores_model_combination:
                scores_model_combination[cur_model_combination] = []
            scores_model_combination[cur_model_combination].append(zscores)

        corr_list = []

        for model_combination, score_list in scores_model_combination.items():
            n = len(score_list)
            if n <= 1:
                continue
            for i, j in combinations(list(range(n)), r=2):
                s1 = score_list[i]
                s2 = score_list[j]
                corr = MTurkStatistics.correlation(s1, s2)
                if pd.isna(corr):
                    continue
                corr_list.append(corr)
        return corr_list

    def _get_corr_between_scores_by_pd(self, pd_sys_score):
        scoretypes = [e for e in pd_sys_score.columns if e not in ['model', 'N']]
        score_dict = {
            e: pd_sys_score[e].values
            for e in scoretypes
        }
        n = len(scoretypes)
        corr_array = np.diag([np.nan for _ in range(n)])
        for i in range(n):
            scoretype1 = scoretypes[i]
            score1 = score_dict[scoretype1]
            for j in range(n):
                if i == j:
                    continue
                scoretype2 = scoretypes[j]
                score2 = score_dict[scoretype2]

                if i < j:
                    corr = MTurkStatistics.correlation(score1, score2, method='p')
                    corr_array[i][j] = corr
                else:
                    # spearman
                    corr = MTurkStatistics.correlation(score1, score2, method='s')
                    corr_array[i][j] = corr
        pd_corr = pd.DataFrame()
        pd_corr[""] = scoretypes
        for i, e in enumerate(scoretypes):
            pd_corr[e] = corr_array[:, i]
        return pd_corr

    def get_corr_between_scores(self):
        pd_system_scores = self.pd_system_scores
        self.pd_corr_scores = self._get_corr_between_scores_by_pd(pd_system_scores)

        pd_system_raw_scores = self.pd_system_raw_scores
        self.pd_corr_raw_scores = self._get_corr_between_scores_by_pd(pd_system_raw_scores)

    def savefiles(self):
        savedir = self.savedir
        savedir.mkdir(exist_ok=True)
        self.draw_sigtest_figure()
        self.draw_rater_agreement_figure()

        # save correlation between z scores as csv&Excel
        f_csv_corr_scores = savedir.joinpath('corr_between_scores.csv')
        self.pd_corr_scores.to_csv(f_csv_corr_scores, index=False)
        f_excel_corr_scores = savedir.joinpath('corr_between_scores.xlsx')
        self.pd_corr_scores.to_excel(f_excel_corr_scores, index=False)

        # save correlation between raw scores as csv&Excel
        f_csv_corr_raw_scores = savedir.joinpath('corr_between_raw_scores.csv')
        self.pd_corr_raw_scores.to_csv(f_csv_corr_raw_scores, index=False)
        f_excel_corr_raw_scores = savedir.joinpath('corr_between_raw_scores.xlsx')
        self.pd_corr_raw_scores.to_excel(f_excel_corr_raw_scores, index=False)

    @staticmethod
    def model_mapping(modelnames):
        models = []
        model_prefixes = {}
        for e in modelnames:
            if e.endswith("_p"):
                model_prefix = e[:-2]
            else:
                model_prefix = e
            if model_prefix not in models:
                models.append(model_prefix)
            model_prefixes[e] = model_prefix
        mappings = {
            e: chr(ord("A") + i)
            for i, e in enumerate(models)
        }
        model_mappings = {
            k: k.replace(v, mappings[v])
            for k, v in model_prefixes.items()
        }
        model_mappings = {k: v.replace('_p', r"$_p$") for k, v in model_mappings.items()}
        return model_mappings

    @staticmethod
    def sig_test(first, second):
        result = mannwhitneyu(first, second, alternative='greater', method="auto")
        pvalue = result.pvalue
        return pvalue

    @staticmethod
    def correlation(arr1, arr2, method='pearson'):
        """
        method: pearson (p), spearman (s) or kendall (k)
        """
        methods = {
            "pearson": pearsonr,
            "p": pearsonr,
            "spearman": spearmanr,
            "s": spearmanr,
            "kendall": kendalltau,
            "k": kendalltau,
        }
        a1 = np.array(arr1)
        a2 = np.array(arr2)
        corr = methods[method](a1, a2)
        if method in ['p', 'pearson']:
            return corr.statistic
        else:
            return corr.correlation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', "--dir", type=str, required=True,
                        help="directory path")

    args, _ = parser.parse_known_args()
    dpath = args.dir
    mturk_stats = MTurkStatistics(dpath)
    mturk_stats.process()
    mturk_stats.savefiles()


if __name__ == '__main__':
    main()
