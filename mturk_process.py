import argparse
import pandas as pd
import numpy as np
import json
from tqdm.auto import tqdm
from pathlib import Path
import random
from scipy.stats import (
    wilcoxon,ranksums,mannwhitneyu,
    pearsonr,spearmanr,kendalltau,
)
import scipy.stats as stats


class ProcessJSONFile:
    def __init__(self,file):
        fpath = Path(file).absolute().resolve()
        self._preprocessing(fpath)
        
        
    def _preprocessing(self,fpath):
        self.file = f"{fpath}"
        print(f'init file {fpath}')
        self.fpath = fpath
        self.savedir = fpath.parent.joinpath("Results")
        self.qc_threshold = 0.05
        with self.fpath.open() as f:
            js_data = json.load(f)
        self.metadata = js_data['metadata']
        self.qc_model = self.metadata['qc-model']
        self.modelnames = self.metadata['model']
        self.meta_score = self.metadata['score']
        self.model_ranking = self.metadata.get('ranking',None)
        self.scorenames = list(self.meta_score.keys())
        self.sorted_scores = self.metadata.get('sorted_scores',self.scorenames)
        self._process_metascore()
        self.js_data = js_data['data']
        self.process_js_data_to_pd()
        self.workerIDs = np.unique(self.pd_data['workerID'].values)
        
    
    def _process_metascore(self):
        positive_scores = []
        negative_scores = []
        qc_scores = []
        score_max = {}
        for scorename, attributes in self.meta_score.items():
            score_max[scorename] = attributes['max']
            if attributes['positive']:
                positive_scores.append(scorename)
            else:
                negative_scores.append(scorename)
            if attributes['qc']:
                qc_scores.append(scorename)
        self.positive_scores = positive_scores
        self.negative_scores = negative_scores
        self.qc_scores = qc_scores
        self.score_max = score_max
    
    def process(self):
        self.get_worker_std_mean()
        self.update_pd_data_with_zscores()
        self.quality_control_by_workers()
        self.compute_passrates()
        self.get_model_scores()
        self.duration_data()
        self.get_model_metadata()
        
    

    def process_js_data_to_pd(self):
        js_data = self.js_data
        pd_lines = []
        reverse = set()
        for e in js_data:
            hitID = e['hit']
            workerID = e['worker']
            assignID = e['assignment']
            result = e['result']
            for dial_data in result:
                model = dial_data['model']
                cur_line = {
                    "model": model,
                }
                score = dial_data['score']
                for scoretype,scorevalue in score.items():
                    if scoretype in self.positive_scores:
                        cur_line[scoretype] = scorevalue
                    else:
                        n_max = self.score_max[scoretype]
                        cur_line[scoretype] = n_max-scorevalue
                        reverse.add(scoretype)
                cur_line.update(
                    {
                        'hitID': hitID,
                        'workerID': workerID,
                        'assignID': assignID,
                    }
                )
                pd_lines.append(cur_line)
        pd_data = pd.DataFrame(pd_lines)
        reverse  = list(reverse)
        reverse = ', '.join(reverse)
        print(f"reversed: [{reverse}]")
        self.pd_data = pd_data
            
        

    def get_std_mean(self, scores):
        std_mean = {
            "mean": np.mean(scores),
            "std": np.std(scores,ddof=1),
        }
        return std_mean
    
    def get_worker_std_mean(self):
        workerIDs = self.workerIDs
        pd_data = self.pd_data
        scorenames = self.scorenames
        worker_std_mean = {}
        for workerID in workerIDs:
            pd_worker = pd_data[pd_data['workerID']==workerID]
            cur_scores = pd_worker[scorenames].values.flatten()
            worker_std_mean[workerID] = self.get_std_mean(cur_scores)
        self.worker_std_mean = worker_std_mean
    
    def update_pd_data_with_zscores(self):
        pd_data = self.pd_data
        epsilon = 0.000001
        stds = []
        means = []
        for _,row in pd_data.iterrows():
            workerID = row['workerID']
            std = self.worker_std_mean[workerID]['std']
            mean = self.worker_std_mean[workerID]['mean']
            stds.append(std)
            means.append(mean)
        stds = np.array(stds)+epsilon
        std_np = stds.reshape(-1,1)
        std_np = np.repeat(std_np,repeats=len(self.scorenames),axis=1)
        
        
        means = np.array(means)
        mean_np = means.reshape(-1,1)
        mean_np = np.repeat(mean_np,repeats=len(self.scorenames),axis=1)
        
        rawscores = pd_data[self.scorenames].values
        zscores = (rawscores-mean_np)/std_np
        raw_scoretypes = [f"raw_{e}" for e in self.scorenames]
        pd_data['std'] = stds
        pd_data['mean'] = means
        pd_data[self.scorenames] = zscores
        pd_data[raw_scoretypes] = rawscores
        
        avg_zscores = np.mean(zscores,axis=1).reshape(-1,1)
        
        pd_data['z'] = avg_zscores
        avg_rawscores = np.mean(rawscores,axis=1).reshape(-1,1)
        pd_data['raw'] = avg_rawscores
        
        self.pd_data = pd_data
            

    def quality_control(self,qc_scores,model_scores):
        result = mannwhitneyu(qc_scores,model_scores,alternative='less',method="auto")
        # result = ranksums(qc_scores,model_scores,alternative='less')
        pvalue = result.pvalue
        return pvalue
    
    def quality_control_by_workers(self):
        worker_pvalues = {}
        passed_workers = []
        failed_workers = []
        for workerID in self.workerIDs:
            worker_std = self.worker_std_mean[workerID]['std']
            worker_mean = self.worker_std_mean[workerID]['mean']
            if worker_std == 0:
                worker_pvalues[workerID] = 1.0
                failed_workers.append(workerID)
                continue
                
            cur_pd = self.pd_data[self.pd_data['workerID']==workerID]
            qc_data = cur_pd[cur_pd['model']==self.qc_model]
            model_data= cur_pd[cur_pd['model'].isin(self.modelnames)]
            
            qc_scores = qc_data[self.qc_scores].values.flatten()
            model_scores = model_data[self.qc_scores].values.flatten()
            
            pvalue = self.quality_control(qc_scores,model_scores)
            worker_pvalues[workerID] = pvalue
            if pd.isna(pvalue) or pvalue >= self.qc_threshold:
                failed_workers.append(workerID)
            else:
                passed_workers.append(workerID)
            
        self.worker_pvalues = worker_pvalues
        self.failed_workers = failed_workers
        self.passed_workers = passed_workers
        
        pd_data = self.pd_data
        
        pd_passed = pd_data[pd_data['workerID'].isin(passed_workers)]
        passed_assigns = pd_passed['assignID'].values
        passed_assigns = np.unique(passed_assigns).tolist()
        passed_hits = pd_passed['hitID'].values
        passed_hits = np.unique(passed_hits).tolist()
        self.pd_passed = pd_passed
        self.passed_assigns = passed_assigns
        self.passed_hits = passed_hits
        
        pd_failed = pd_data[pd_data['workerID'].isin(failed_workers)]
        failed_assigns = pd_failed['assignID'].values
        failed_assigns = np.unique(failed_assigns).tolist()
        failed_hits = pd_failed['hitID'].values
        failed_hits = np.unique(failed_hits).tolist()
        self.pd_failed = pd_failed
        self.failed_assigns = failed_assigns
        self.failed_hits = failed_hits
        
        
    
    def get_model_scores(self):
        pd_passed = self.pd_passed
        
        if self.model_ranking is None:
            modelnames = self.modelnames
        else:
            modelnames = self.model_ranking
        data_model_scores = []
        data_model_raw_scores = []
        for modelname in modelnames:
           
            pd_model = pd_passed[pd_passed['model']==modelname]
            n = len(pd_model)*len(self.sorted_scores)
            model_dict = {
                "model": modelname,
                "N": n,
            }
            model_dict_raw = {
                "model": modelname,
                "N": n,
            }
            # z score
            zscores = pd_model[self.sorted_scores].values
            mean_zscores = np.mean(zscores,axis=0).tolist()
            mean_zscore_dict = dict(zip(self.sorted_scores,mean_zscores))
            overall_zscore = np.mean(zscores)
            model_dict['z'] = overall_zscore
            model_dict.update(mean_zscore_dict)
            data_model_scores.append(model_dict)
            
            # raw score 
            rawscores = pd_model[[f"raw_{e}" for e in self.sorted_scores]].values
            mean_rawscores = np.mean(rawscores,axis=0).tolist()
            mean_rawscore_dict = dict(zip(self.sorted_scores,mean_rawscores))
            overall_rawscore = np.mean(rawscores)
            model_dict_raw['raw'] = overall_rawscore
            model_dict_raw.update(mean_rawscore_dict)
            data_model_raw_scores.append(model_dict_raw)
            
        pd_model_scores = pd.DataFrame(data_model_scores)
        pd_model_raw_scores = pd.DataFrame(data_model_raw_scores)

        # if no model ranking specified, rank by z scores.
        if self.model_ranking is None:
            pd_model_scores = pd_model_scores.sort_values(by=['z'],ascending=False).reset_index(drop=True)
            
            # sort raw by z
            pd_model_raw_scores = pd.concat([
                pd_model_raw_scores[pd_model_raw_scores['model']==m] 
                for m in pd_model_scores['model'].values
            ]).reset_index(drop=True)
        ranked_models = pd_model_scores['model'].values.tolist()
        self.ranked_models = ranked_models
        self.pd_model_scores = pd_model_scores
        self.pd_model_raw_scores = pd_model_raw_scores
    

        
    def compute_passrates(self):
        passrate_data = []
    
        n_passed_hit = len(self.passed_hits)
        n_all_hit = n_passed_hit+len(self.failed_hits)
        passrate_hit = n_passed_hit/n_all_hit
        hit_dict = {
            "": "HIT",
            "passed": n_passed_hit,
            "total": n_all_hit,
            "passrate": passrate_hit,
        }
        passrate_data.append(hit_dict)
        
        n_passed_worker = len(self.passed_workers)
        n_all_worker = n_passed_worker+len(self.failed_workers)
        passrate_worker = n_passed_worker/n_all_worker
        worker_dict = {
            "": "Worker",
            "passed": n_passed_worker,
            "total": n_all_worker,
            "passrate": passrate_worker,
        }
        passrate_data.append(worker_dict)
        self.pd_passrate = pd.DataFrame(passrate_data)
        
    def duration_data(self):
        passed_durations = []
        failed_durations = []
        
        for e in self.js_data:
            worker = e['worker']
            duration = int(e['duration in seconds'])
            if worker in self.passed_workers:
                passed_durations.append(duration)
            else:
                failed_durations.append(duration)
        all_durations = passed_durations+failed_durations
        
        passed_avg_duration_sconds = np.mean(passed_durations)
        failed_avg_duration_sconds = np.mean(failed_durations)
        all_avg_duration_sconds = np.mean(all_durations)
        
        model_num = len(self.js_data[0]['result']) # get models per hit
        
        duration_data = {
            'passed': passed_avg_duration_sconds/model_num/60,
            'failed': failed_avg_duration_sconds/model_num/60,
            'all': all_avg_duration_sconds/model_num/60,
        } 
        self.pd_duration = pd.DataFrame([duration_data])

    def get_model_metadata(self):
        ranked_models = self.ranked_models
        qc_model = self.qc_model
        model_metadata = {
            "ranked_model": ranked_models,
            "qc_model": qc_model,
        }
        self.model_metadata = model_metadata
        
    def save_files(self):
        savedir = self.savedir
        savedir.mkdir(exist_ok=True)

        # save meta data
        f_model_metadata = savedir.joinpath("model_metadata.json")
        with f_model_metadata.open('w') as f:
            json.dump(self.model_metadata,f,indent=2)
        
        # save worker_ids
        f_passed_worker = savedir.joinpath("passed_workers.json")
        with f_passed_worker.open('w') as f:
            json.dump(self.passed_workers,f,indent=2)
        f_failed_worker = savedir.joinpath("failed_worker.json")
        with f_failed_worker.open('w') as f:
            json.dump(self.failed_workers,f,indent=2)
            
        # save assignment_ids
        f_passed_assign = savedir.joinpath("passed_assignment.json")
        with f_passed_assign.open('w') as f:
            json.dump(self.passed_assigns,f,indent=2)
        f_failed_assign = savedir.joinpath("failed_assignment.json")
        with f_failed_assign.open('w') as f:
            json.dump(self.failed_assigns,f,indent=2)
        
        # save hit_ids
        f_passed_hit = savedir.joinpath("passed_hit.json")
        with f_passed_hit.open('w') as f:
            json.dump(self.passed_hits,f,indent=2)
        f_failed_hit = savedir.joinpath("failed_hit.json")
        with f_failed_hit.open('w') as f:
            json.dump(self.failed_hits,f,indent=2)
            
        # save data as csv
        f_csv_all = savedir.joinpath("system_scores_all.csv")
        self.pd_data.to_csv(f_csv_all,index=False)
        f_csv_passed = savedir.joinpath("system_scores_passed.csv")
        self.pd_passed.to_csv(f_csv_passed,index=False)
        f_csv_failed = savedir.joinpath("system_scores_failed.csv")
        self.pd_failed.to_csv(f_csv_failed,index=False)
        
        # save pass rate as Excel&csv
        f_excel_passrate = savedir.joinpath("passrate.xlsx")
        self.pd_passrate.to_excel(f_excel_passrate,index=False)
        f_csv_passrate = savedir.joinpath("passrate.csv")
        self.pd_passrate.to_csv(f_csv_passrate,index=False)
        
        # save model z scores as Excel&csv
        f_excel_model_score = savedir.joinpath("system_scores.xlsx")
        self.pd_model_scores.to_excel(f_excel_model_score,index=False)
        f_csv_model_score = savedir.joinpath("system_scores.csv")
        self.pd_model_scores.to_csv(f_csv_model_score,index=False)
        
        # save model raw scores as Excel&csv
        f_excel_model_raw_score = savedir.joinpath("system_raw_scores.xlsx")
        self.pd_model_raw_scores.to_excel(f_excel_model_raw_score,index=False)
        f_csv_model_raw_score = savedir.joinpath("system_raw_scores.csv")
        self.pd_model_raw_scores.to_csv(f_csv_model_raw_score,index=False)
        
        
        # save duration raw scores as Excel&csv
        f_excel_duration = savedir.joinpath("duration.xlsx")
        self.pd_duration.to_excel(f_excel_duration,index=False)
        f_csv_duration = savedir.joinpath("duration.csv")
        self.pd_duration.to_csv(f_csv_duration,index=False)
        
    
        
    @staticmethod
    def correlation(arr1,arr2,method='pearson'):
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
        corr = methods[method](a1,a2)
        return corr.statistic

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', "--file", type=str,required=True,
                        help="json file")
    
    args,_ = parser.parse_known_args()
    file = args.file
    # print(file)
    process_js = ProcessJSONFile(file)
    process_js.process()
    process_js.save_files()

if __name__ == '__main__':
    main()