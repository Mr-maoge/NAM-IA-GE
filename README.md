# Paper
Gene-environment Interaction Analysis via Pathological Imaging-assisted Neural Additive Model (forthcoming)

# Maintainer
Jingmao Li,  [jingmao.li@yale.edu](jingmao.li@yale.edu)  

# Files and functions
* `Method.py`  
    This file contains the class for the proposed Neural Additive Model, the class for model training  
    Main Classes and functions:
    * `Class Model_single_modal`  
        Neural Additive Model for G--E interaction model 
    * `Class Model_multi_modal`  
        Neural Additive Model for G--E interaction model (used in the joint learning)
    * `Class Trainer`  
        Trainer for the network  

* `DGP.py`  
    This file contains the function for similation data generation  
    Main functions:
    * `generate_data`  
        The function used to generate simulation data
* `sim_funcs.py`  
    This file contains the function for conducting similation (for the proposed method)
    * `sim_func_exp1`
        This function used in simulation study 
    * `sim_multi_times`
        The function used to run simulations for multiple times
* `sim_exps.py`  
    This is the main file, which conducts simulations for various settings. The parallel runing is used to fasten the computation.
* `sim_schemes5.xlsx`  
    The table containing the simulation settings. 
    * case="nonliner3": Example 1 in paper
    * case="nonliner1": Example 2 in paper
    * case="nonliner2": Example 3 in paper
    * case="linear": Example 4 in paper

* `proxy.py`  
    The proximal functions used in model fitting.
* `utils.py`  
    Some util functions used in the project. 

# Usage
* Run `sim_exps.py` to get the summary of simulation results (based on 100 replicates) of the proposed methods. The resulting table are stored in folder `./result/`.

