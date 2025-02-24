import copy
from DGP import generate_data, data_to_torch
import pandas as pd
import multiprocessing as mp
from utils import *
import pickle
import os
from Method import *
import time


def sim_func_exp(seed_=1, n=1000, r=100, p=100, q=5,
                  case="linear", H_type="diagnal",
                  n_main=12, n_interactions=14,
                  rho=0.3, metrics=["TP_main", "FP_main", "TP_inter", "FP_inter", "metric"],
                  metric_name="bic",
                  **kwargs):
    """
    This function is used in simulation study
    :param seed_: int, random seed
    :param n: int, sample size
    :param r: int, number of G variables
    :param p: int, number of I variables
    :param q: int, number of E variables
    :param case: str, "linear", "nonlinear1", "nonlinear2", or "nonlinear3"
    :param H_type: str, "diagnal", "band1", "band2", "milky_way", or "nonlinear"
    :param n_main: int, number of significant main G effects in true model
    :param n_interactions: int, number of significant interaction effects in true model
    :param rho: float, the parameter of the AR structure covariance matrix of G variables (and E variables)
    :param metrics: list of str, evaluation metrics for output
    :param metric_name: the metric for selecting the best hyperparameters, including "bic", "aic", "metric"
    """

    #### Generate data ####
    tra_data, val_data, tes_data, coef_true = generate_data(n=n, n_val=200, n_tes=4000, r=r, p=p, q=q,
                                                            case=case, H=None, H_type=H_type,
                                                            n_main=n_main, n_interactions=n_interactions,
                                                            censor_rate=0.3,
                                                            rho=rho, seed=seed_
                                                            )
    tra_X, tra_Y = data_to_torch(tra_data)
    val_X, val_Y = None, None
    tes_X, tes_Y = data_to_torch(tes_data)

    result = {}
    computation_time = {}
    best_tunings = {}

    #### init models ####
    ## G single model init (G_model_single)
    start_time = time.time()
    model = Model_single_modal(r=r, q=q, is_linear=False,
                                G_name="Z", E_name="E",
                                dims1=(1, 5, 5), dims2=(1, 5, 5), dims3=(1, 5, 5)
                                )
    trainer_single = Trainer(model, penalty="our_mcp")
    trainer_single.train(tra_X, tra_Y,
                            kwarg_prox={}, kwarg_loss={}, lr=0.01, maxit=20, tol=1e-3, init_state_dict=None,
                            val_X_dic=None, val_Y_dic=None,
                            prox_it=100, min_it=1, eval_it=100, early_stop_round=100, use_refit=True
                            )
    G_model_single = quick_deepcopy(trainer_single.model)
    res_coef = G_model_single.evaluate_coef(coef_true)
    res_test = G_model_single.evaluate_test(tes_X, tes_Y)
    end_time = time.time()
    computation_time_G_init = end_time - start_time  
        
    ## I single model init (I_model_single)
    start_time = time.time()
    model = Model_single_modal(r=p, q=q, is_linear=False,
                                G_name="X", E_name="E",
                                dims1=(1, 5, 5), dims2=(1, 5, 5), dims3=(1, 5, 5)
                                )
    trainer_alt1 = Trainer(model, penalty="our_mcp")
    trainer_alt1.train(tra_X, tra_Y,
                        kwarg_prox={}, kwarg_loss={}, lr=0.01, maxit=20, tol=1e-3, init_state_dict=None,
                        val_X_dic=None, val_Y_dic=None,
                        prox_it=100, min_it=1, eval_it=100, early_stop_round=100, use_refit=True
                        )
    I_model_single = quick_deepcopy(trainer_alt1.model)
    res_test = I_model_single.evaluate_test(tes_X, tes_Y)
    end_time = time.time()
    computation_time_I_init = end_time - start_time  

    #### Train the Proposed method ####
    # Setup the model
    start_time = time.time()
    F_mat = estimate_F(tra_X["Z"], tra_X["X"], type="spearman", thresh=0.1)
    G_model = quick_deepcopy(G_model_single)
    I_model = quick_deepcopy(I_model_single)
    model = Model_multi_modal(r=r, p=p, q=q, is_linear=False,
                                G_name="Z", E_name="E", I_name="X",
                                G_model=G_model, I_model=I_model,
                                F_mat=F_mat, cr_type=1
                                )
    ### Train the model
    trainer_multi = Trainer(model, penalty="our_mcp")
    # set the candidate hyperparameters
    kwarg_prox_set = {"lam1_1": [0.04], "lam1_2": [0.03]}   
    kwarg_loss_set = {"gamma": [0.01]}
    lrs = [0.02]
    # model training
    trainer_multi.train_path(tra_X, tra_Y,
                                kwarg_prox_set=kwarg_prox_set, kwarg_loss_set=kwarg_loss_set, lrs=lrs, tol=1e-4, maxit=50,
                                init_state_dict=None,
                                val_X_dic=val_X, val_Y_dic=val_Y,
                                metric_name=metric_name,
                                prox_it=10, min_it=10, eval_it=10, early_stop_round=10
                                )
    # evaluation for G-E model
    res_coef = trainer_multi.model.G_model.evaluate_coef(coef_true)
    res_test = trainer_multi.model.G_model.evaluate_test(tes_X, tes_Y)
    end_time = time.time()
    computation_time = (end_time - start_time) + \
                        computation_time_G_init + computation_time_I_init
    result["Proposed"] = {"Time": computation_time, **res_coef, **res_test}
    best_tunings["Proposed"] = trainer_multi.best_tunings

    return pd.DataFrame(result).loc[metrics, :].T, best_tunings


def sim_multi_times(seed_lst, sim_func, kwargs, cores=None, folder_path=None, label=""):
    """
    Run simulation multiple times
    :param seed_lst: list of int, random seeds
    :param sim_func: function, simulation function
    :param kwargs: dict, parameters for simulation function
    :param cores: int, number of cores used
    :param folder_path: str, the path to save the results
    :param label: str, the label for the results
    """
    if cores is None:
        cores = min(mp.cpu_count() - 2, len(list(seed_lst)))
    print(f"{cores} cores used.")
    p = mp.Pool(cores)
    res = []
    for seed_ in seed_lst:
        kw = copy.deepcopy(kwargs)
        kw["seed_"] = seed_
        out = p.apply_async(sim_func, kwds=kw)
        res.append(out)
    p.close()
    p.join()

    res_df_lst, res_tunings_lst = list(zip(*[out.get() for out in res]))
    p.terminate()

    res_df = merge_DataFrame(list(res_df_lst))
    res_tunings = merge_dics(list(res_tunings_lst))

    print(res_df)
    print(res_tunings)

    if folder_path is not None:
        res_df.to_csv(os.path.join(folder_path, f"{label}_result.csv"))
        pickle.dump(res_tunings, open(os.path.join(folder_path, f"{label}_tunings.pkl"), "wb") )

