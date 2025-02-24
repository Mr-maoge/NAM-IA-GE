import sys
sys.path.append("Z:/python_packages")
import random
import numpy as np
from collections import defaultdict
import logging
import pandas as pd
import pickle
import torch

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def sigmoid(z):
    return 1/(1 + np.exp(-z))


def get_settings(sim_schems, idx):
    all_setting = sim_schems.to_dict()
    setting = {}
    for k, v in all_setting.items():
        setting[k] = v[idx]
    return setting


def quick_deepcopy(obj):
    return pickle.loads(pickle.dumps(obj))


def merge_dics(dic_lst):
    res_dic = defaultdict(list)
    for dic in dic_lst:
        for k, v in dic.items():
           res_dic[k].append(v)
    return res_dic


def merge_DataFrame(df_lst, index_cols=None, mean_only=False):
    if index_cols is None:
        for i, df in enumerate(df_lst):
            df_lst[i] = df.reset_index()
        index_cols = ["index"]
    df_tot = pd.concat(df_lst, axis=0)
    cols = [col for col in df_tot.columns if col not in index_cols]
    gb = df_tot.groupby(index_cols)
    res_mean, res_std = gb[cols].mean(), gb[cols].std()
    if not mean_only:
        output = res_mean.applymap(lambda x:"%.3f"%x) + "(" + res_std.applymap(lambda x:"%.3f"%x) + ")"
    else:
        output = res_mean
    return output


def creater_logger(logger_file):
    logger = logging.getLogger('mylogger')
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(logger_file)
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('[%(asctime)s][%(levelname)s] ## %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def logrange(start, end, num):
    return np.exp(np.linspace(np.log(start), np.log(end), num))


# util funcs for output result

def __mean(x):
    try:
        r = float(re.match("(\d+\.\d+?)\((.*?)\)", x).group(1))
    except AttributeError:
        r = np.nan
    return r


def __std(x):
    try:
        r = float(re.match("(\d+\.\d+?)\((.*?)\)", x).group(2))
    except AttributeError:
        r = np.nan
    return r


def round_digits(s):
    out = "%.3f(%.3f)" % (__mean(s), __std(s))
    out = out.replace("nan", "-")
    return (out)


def mean_std_extract(df, digit=3):
    df_mean = df.applymap(__mean)
    df_std = df.applymap(__std)
    return df_mean, df_std


def concordance_index(risk, time, event):
    assert len(risk) == len(time) == len(event)
    n = len(risk)
    tot = 0
    cnt = 0
    for i in range(n):
        for j in range(i, n):
            if event[i] == 1 and event[j] == 1:
                tot += 1
                if time[i] == time[j]:
                    cnt += 1
                elif time[i] < time[j]:
                    cnt += int(risk[i]>risk[j])
                else:
                    cnt += int(risk[i]<risk[j])
            elif event[i] == 1:
                if time[i] == time[j]:
                    tot += 1
                    cnt += 1
                elif time[i] < time[j]:
                    tot += 1
                    cnt += int(risk[i]>risk[j])
            elif event[j] == 1:
                if time[i] == time[j]:
                    tot += 1
                    cnt += 1
                elif time[i] > time[j]:
                    tot += 1
                    cnt += int(risk[i]<risk[j])

    return cnt / tot


def summarize_coefs_result(df_tab, r, q):
    df_num_main = df_tab.iloc[:r, :].apply(np.sum, axis=0)
    df_num_inter = df_tab.iloc[r:(r*(q+1)), :].apply(np.sum, axis=0)
    return pd.DataFrame(dict(main=df_num_main, inter_IE=df_num_inter))

