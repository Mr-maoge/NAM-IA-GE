from sim_funcs import *

if __name__ == "__main__":
    # =============================================
    sim_schemes = pd.read_excel("sim_schemes5.xlsx") # the simulation schemes
    rep_times = 20                                   # number of repetitions
    metric_name = "bic"                              # the metric for selecting the best hyperparameters, including "bic", "aic", "metric"
    folder = f"./result/"                 # the folder to save the results

    # create the logger
    logger_file = os.path.join(folder, "training.log") 
    if not os.path.exists(folder):
        os.mkdir(folder)
    logger = creater_logger(logger_file)
    # =============================================
    # run the simulation
    for i in range(sim_schemes.shape[0]):
        logger.info(f"==== setting {i} ====")
        setting = get_settings(sim_schemes, i)
        metrics = ["TP_main", "FP_main", "TP_inter", "FP_inter", "metric",
                   "TPR_main", "FPR_main", "TPR_inter", "FPR_inter", "Time"
                   ]
        setting["metrics"] = metrics
        setting["metric_name"] = metric_name
        sim_multi_times(range(rep_times), sim_func_exp, setting, cores=4, folder_path=folder, label=str(i))

