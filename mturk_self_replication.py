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
from collections import OrderedDict
import copy, os


class MturkSelfRepicaltion:

    def __init__(self, dir_run1, dir_run2):
        self.dir_r1 = Path(dir_run1).absolute().resolve()
        self.dir_r2 = Path(dir_run2).absolute().resolve()
        self.savedir = self.dir_r1.joinpath('corr-r1r2')
        self._preprocess()

    def _preprocess(self):
        f_score_r1 = self.dir_r1.joinpath('system_scores.csv')
        pd_score_r1 = pd.read_csv(f_score_r1)
        f_raw_score_r1 = self.dir_r1.joinpath('system_raw_scores.csv')
        pd_raw_score_r1 = pd.read_csv(f_raw_score_r1)

        f_score_r2 = self.dir_r2.joinpath('system_scores.csv')
        pd_score_r2 = pd.read_csv(f_score_r2)
        f_raw_score_r2 = self.dir_r2.joinpath('system_raw_scores.csv')
        pd_raw_score_r2 = pd.read_csv(f_raw_score_r2)

        model_ranking = pd_score_r1.model.values
        pd_raw_score_r1 = MturkSelfRepicaltion.sort_pd_by_model_rank(pd_raw_score_r1, model_ranking)
        pd_score_r2 = MturkSelfRepicaltion.sort_pd_by_model_rank(pd_score_r2, model_ranking)
        pd_raw_score_r2 = MturkSelfRepicaltion.sort_pd_by_model_rank(pd_raw_score_r2, model_ranking)

        z_scoretypes = [e for e in pd_score_r1.columns.values if e not in ["model", "N"]]
        raw_scoretypes = [e for e in pd_raw_score_r1.columns.values if e not in ["model", "N"]]

        self.model_ranking = model_ranking
        self.z_scoretypes = z_scoretypes
        self.raw_scoretypes = raw_scoretypes
        self.pd_score_r1 = pd_score_r1
        self.pd_raw_score_r1 = pd_raw_score_r1
        self.pd_score_r2 = pd_score_r2
        self.pd_raw_score_r2 = pd_raw_score_r2

    @staticmethod
    def sort_pd_by_model_rank(pd_score, model_ranking):
        pd_new = pd.concat([pd_score[pd_score['model'] == e] for e in model_ranking])
        pd_new = pd_new.reset_index(drop=True)
        return pd_new

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

    def process(self):
        self.compute_corr_r1r2()

    def savefiles(self):
        savedir = self.savedir
        savedir.mkdir(exist_ok=True)
        # save correlations of zscores as csv&Excel
        f_csv_corr_r1r2 = savedir.joinpath("corr_zscore_r1r2.csv")
        self.pd_z_corr_r1r2.to_csv(f_csv_corr_r1r2, index=False)
        f_excel_corr_r1r2 = savedir.joinpath("corr_zscore_r1r2.xlsx")
        self.pd_z_corr_r1r2.to_excel(f_excel_corr_r1r2, index=False)

        # save correlations of rawscores as csv&Excel
        f_csv_corr_raw_r1r2 = savedir.joinpath("corr_rawscore_r1r2.csv")
        self.pd_raw_corr_r1r2.to_csv(f_csv_corr_raw_r1r2, index=False)
        f_excel_corr_raw_r1r2 = savedir.joinpath("corr_rawscore_r1r2.xlsx")
        self.pd_raw_corr_r1r2.to_excel(f_excel_corr_raw_r1r2, index=False)

    def compute_corr_r1r2(self):
        pd_score_r1 = self.pd_score_r1
        pd_raw_score_r1 = self.pd_raw_score_r1
        pd_score_r2 = self.pd_score_r2
        pd_raw_score_r2 = self.pd_raw_score_r2
        corr_types = ["pearson", "spearman", "kendall"]

        corr_z_r1r2 = []
        corr_raw_r1r2 = []
        for corr_type in corr_types:
            cur_z_data = {
                "": corr_type,
            }

            for score_type in self.z_scoretypes:
                score1 = pd_score_r1[score_type].values
                score2 = pd_score_r2[score_type].values
                corr = MturkSelfRepicaltion.correlation(score1, score2, method=corr_type)
                cur_z_data[score_type] = corr
            corr_z_r1r2.append(cur_z_data)

            cur_raw_data = {
                "": corr_type,
            }
            for raw_score_type in self.raw_scoretypes:
                raw_score1 = pd_raw_score_r1[raw_score_type].values
                raw_score2 = pd_raw_score_r2[raw_score_type].values
                raw_corr = MturkSelfRepicaltion.correlation(raw_score1, raw_score2, method=corr_type)
                cur_raw_data[raw_score_type] = raw_corr
            corr_raw_r1r2.append(cur_raw_data)
        self.pd_z_corr_r1r2 = pd.DataFrame(corr_z_r1r2)
        self.pd_raw_corr_r1r2 = pd.DataFrame(corr_raw_r1r2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d1', "--dpath1", type=str, required=True,
                        help="path to results folder of first run")
    parser.add_argument('-d2', "--dpath2", type=str, required=True,
                        help="path to results folder of second run")
    args, _ = parser.parse_known_args()
    d1 = args.dpath1
    d2 = args.dpath2
    mturk_self_rep = MturkSelfRepicaltion(d1,d2)
    mturk_self_rep.process()
    mturk_self_rep.savefiles()


if __name__ == '__main__':
    main()
