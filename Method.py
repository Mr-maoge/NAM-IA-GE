import sys
sys.path.append("Z:/python_packages")
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, AdamW
from proxy import prox_GE_all
import numpy as np
from utils import *
from lassonet.cox import CoxPHLoss, concordance_index
from itertools import product
from sklearn.linear_model import ElasticNet


def compute_rank_correlation(Z_, X_):
    """
    Compute the rank correlation between Z and X
    :param Z_: [n, r]
    :param X_: [n, p]   
    :return: corr_mat [p, r]
    """
    n = Z_.shape[0]
    Z = Z_.argsort(dim=0).argsort(dim=0).unsqueeze(1) # find the rank along dim=0
    X = X_.argsort(dim=0).argsort(dim=0).unsqueeze(2)

    upper = 6*torch.sum((Z-X)**2, dim=0)
    down = n*(n**2-1)
    corr_mat = 1 - upper / down
    return corr_mat


def compute_F_ARMI(Z_, X_):
    """
    Compute the ARMI matrix F
    :param Z_: [n, r]
    :param X_: [n, p]
    :return: F [p, r]
    """
    Z = Z_.numpy()  # genes [n, r]
    X = X_.numpy()  # imaging features [n, p]
    model = ElasticNet(alpha=0.1)
    model.fit(X, Z)
    F_ = model.coef_.T  # [p, r]
    F_ = torch.from_numpy(F_).float()
    return F_


def estimate_F(Z, X, type="spearman", thresh=0.0, quantile=0.9):
    """
    Estimate person/spearman correlation matrix or the F matrix for ARMI, based on Z and X
    :param Z: [n, r]
    :param X: [n, p]
    :param type: "person", "spearman", "ARMI"
    :param thresh: threshold for the correlation
    :param quantile: quantile for the correlation
    :return: cor_mat [p, r]
    """
    assert type in ["person", "spearman", "ARMI"], '''type must in ["person", "spearman"]'''
    r = Z.shape[1]
    p = X.shape[1]
    if type == "pearson":
        cor_mat = torch.corrcoef(torch.cat([Z, X], dim=1).T)[r:(r + p), :r].abs()
    elif type == "spearman":
        cor_mat = compute_rank_correlation(Z, X).abs()
    elif type == "ARMI":
        cor_mat = compute_F_ARMI(Z, X)
    if quantile:
        thresh = cor_mat.abs().quantile(quantile).item()
    cor_mat[cor_mat.abs() < thresh] = 0
    return cor_mat


class NAM_linear(nn.Module):
    """
    Linear layer for Neural Additive Model (NAM)
    The 'broadcast' mechanisim in torch is used to fasten the computation
    """
    def __init__(self, pre_dims=(10,), in_features=10, out_features=10):
        """
        :param pre_dims: the dimensions of the preordinate dimensions 
        :param in_features: the number of input features
        :param out_features: the number of outputs
        """
        super().__init__()
        self.in_features = in_features
        self.out_feautres = out_features
        dims_weight = list(pre_dims) + [out_features, in_features]
        dims_bias = list(pre_dims) + [out_features, 1]

        self.weight = nn.Parameter(torch.zeros(dims_weight))
        self.bias = nn.Parameter(torch.zeros(dims_bias))
        self._init_xavier()

    def forward(self, X):
        """
        :param X: pre_dims + [in_features]
        :return:  pre_dims + [out_features]
        """
        return self.weight.matmul(X) + self.bias

    def _init_xavier(self):
        """
        Initialize the weight and bias
        """
        val = np.sqrt(6/(self.in_features+self.out_feautres))
        torch.nn.init.uniform_(self.weight, a=-val, b=val)


class SUB_NET(nn.Module):
    def __init__(self, pre_dims=(), *dims):
        """
        :param pre_dims: the dimensions of the preordinate dimensions
        :param dims: the dimensions of the hidden layers
        """
        super().__init__()
        self.layers_linear = nn.ModuleList(
            [NAM_linear(pre_dims, dims[i], dims[i + 1]) for i in range(len(dims)-1)]
        )
        self.output = NAM_linear(pre_dims, dims[-1], 1)

    def forward(self, X):
        """
        :param X: pre_dims + [dims[0]]
        :return: pre_dims + [dims[-1]]
        """
        for layer_linear in self.layers_linear:
            X = layer_linear(X)
            X = F.relu(X)
        return self.output(X)


class Model_single_modal(nn.Module):
    """
    Neural Additive Model for G--E interaction model (single modality)
    """
    def __init__(self, r=100, q=4, is_linear=False,
                 G_name = "X", E_name="E",
                 dims1=(1, 10), dims2=(1, 10), dims3=(1, 10)):
        """
        :param r: the number of G variables
        :param q: the number of E variables  
        :param is_linear: whether the model is linear or not
        :param G_name: the name of G variable in the input dictionary (in the input of `forward` function)
        :param E_name: the name of E variable in the input dictionary (in the input of `forward` function)
        :param dims1: the sizes for layers in the G-subnet (the *dims paramerter for the `SUB_NET` class) 
        :param dims2: the sizes for layers in the E-subnet (the *dims paramerter for the `SUB_NET` class)
        :param dims3: the sizes for layers in the interaction-subnet (the *dims paramerter for the `SUB_NET` class)
        """
        super().__init__()
        self.G_name = G_name
        self.E_name = E_name
        self.r = r
        self.q = q
        self.is_linear = is_linear

        if not is_linear:
            self.main_G_net = SUB_NET((1, self.r), *dims1)
            self.main_E_net = SUB_NET((1, self.q), *dims2)
            self.inter_net = SUB_NET((1, self.r*self.q), *dims3)
        '''
            Coefficients for variable selection: main_G_effects + inter_effects + main_E effects;
            inter_effects: reshape from [q*r] matrix 
        '''
        self.coef = nn.Parameter(torch.ones((r*(q+1)+q))/(r*(q+1)+q)) # parameters for selection layers 
        self.batchnrom = nn.BatchNorm1d(r*(q+1)+q, momentum=0, affine=False, track_running_stats=True)
        self.batchnrom_out = nn.BatchNorm1d(1, momentum=0, affine=False, track_running_stats=True)

    def forward(self, **tra_X_dic):
        """
        :param tra_X_dic: the input dictionary, including the G and E variables, 
                          the G varaibles are in the key `G_name` with dimension [n, r],
                          and the E variables are in the key `E_name` with dimension [n, q]
        :return: pred: the predicted value for the network with dimension [n, 1]
                 pred_sep: the predicted value for each G main effect and its corresponding interaction terms with dimension [n, r]
        """
        X = tra_X_dic[self.G_name]
        E = tra_X_dic[self.E_name]
        W = E.reshape(-1, self.q, 1).matmul(X.reshape(-1, 1, self.r)).reshape(-1, self.r * self.q)
        X_ = X.unsqueeze(-1).unsqueeze(-1)
        E_ = E.unsqueeze(-1).unsqueeze(-1)
        W_ = W.unsqueeze(-1).unsqueeze(-1)
        res = torch.cat([X, W, E], dim=1)
        if not self.is_linear:
            main_G = self.main_G_net(X_).squeeze()
            main_E = self.main_E_net(E_).squeeze()
            inter = self.inter_net(W_).squeeze()
            res = res + torch.cat([main_G, inter, main_E], dim=1) 
        res = self.batchnrom(res)
        res = res * self.coef
        tmp = res[:, :((self.q+1)*self.r)].reshape(-1, self.q+1, self.r)
        pred_sep = tmp.sum(dim=1)
        pred = res.sum(axis=1, keepdim=True)
        return pred, pred_sep

    def loss_func(self, out, Y, N_, **kwargs):
        """
        Loss function for the network
        :param out: the output of the network (`forward` function)
        :param Y: the observed survival time
        :param N_: the observed event indicator
        :return: the loss value (partial likelihood)
        """
        return CoxPHLoss("breslow")(out[0], torch.stack([Y, N_], dim=1))

    def metric_func(self, log_h, Y, N_):
        """
        C-index 
        :param log_h: the predicted log hazard
        :param Y: the observed survival time
        :param N_: the observed event indicator
        :return: the concordance index
        """
        return concordance_index(log_h, Y, N_)

    def prox(self, penalty, **kwargs):
        """
        Proximal operator for the network
        :param penalty: the penalty type
        :param kwargs: the penalty parameters
        """
        lam1 = kwargs.get("lam1", None)
        lam2 = kwargs.get("lam2", None)
        coef_prox = prox_GE_all(self.coef, r=self.r, q=self.q,
                                      lam1=lam1, lam2=lam2,
                                      penalty=penalty
                                     )
        self.coef.data = coef_prox.data

    def evaluate_test(self, tes_X_dic=None, tes_Y_dic=None, **kwargs):
        """
        Evaluate the network on the test set
        :param tes_X_dic: the input dictionary for the test set (same as the `tra_X_dic` in the `forward` function)
        :param tes_Y_dic: the output dictionary for the test set (same as the `tra_Y_dic` in the `loss_func` function)
        :return: the evaluation results, including the loss value and the metric value
        """
        if tes_X_dic is None:
            return {"loss":1, "metric": 0}
        self.eval()
        out = self(**tes_X_dic)
        n = out[0].shape[0]
        res = {}
        res["loss"] = self.loss_func(out, **tes_Y_dic).item()
        num_nonzero = (self.coef != 0).sum().item()
        res["aic"] = n*res["loss"] + 2*num_nonzero
        res["bic"] = n*res["loss"] + np.log(n)*num_nonzero
        res["metric"] = self.metric_func(out[0], **tes_Y_dic)
        return res

    def evaluate_coef(self, coef_true_):
        """
        Evaluate the network on the variable selection
        :param coef_true_: the true significance for variables
        :return: the evaluation results, including the TP, FP, TPR, FPR for main effects and interaction effects
        """
        r, q = self.r, self.q
        res = {}
        coef_all = self.coef.detach().numpy()
        # main  effects
        coef = coef_all[:r]
        coef_true = coef_true_[:r]
        res["TP_main"] = ((coef!=0)*(coef_true!=0)).sum()
        res["FP_main"] = ((coef!=0)*(coef_true==0)).sum()
        res["TPR_main"] = res["TP_main"] / (coef_true!=0).sum()
        res["FPR_main"] = res["FP_main"] / (coef_true==0).sum()

        # inter effects
        coef = coef_all[r:(r*(q+1))]
        coef_true = coef_true_[r:(r*(q+1))]
        res["TP_inter"] = ((coef != 0) * (coef_true != 0)).sum()
        res["FP_inter"] = ((coef != 0) * (coef_true == 0)).sum()
        res["TPR_inter"] = res["TP_inter"] / (coef_true != 0).sum()
        res["FPR_inter"] = res["FP_inter"] / (coef_true == 0).sum()

        return res


class Model_multi_modal(nn.Module):
    """
    Neural Additive Model for G--E interaction model (multi-modality)
    """
    def __init__(self, r=100, p=100, q=4, is_linear=False,
                 G_name = "Z", E_name="E", I_name="X",
                 G_model = None, I_model = None,
                 dims1=(1, 10), dims2=(1, 10), dims3=(2, 10), dropout=None,
                 F_mat=None,
                 cr_type=0):
        """
        :param r: the number of G variables
        :param p: the number of I variables
        :param q: the number of E variables
        :param is_linear: whether the model is linear or not
        :param G_name: the name of G variable in the input dictionary (in the input of `forward` function)
        :param E_name: the name of E variable in the input dictionary (in the input of `forward` function)
        :param I_name: the name of I variable in the input dictionary (in the input of `forward` function)
        :param G_model: the model for the G--E interaction model (NAM)
        :param I_model: the model for the I--E interaction model (NAM)
        :param dims1: the sizes for layers in the G/I-subnet (the *dims paramerter for the `SUB_NET` class), not used if `G_model` or `I_model` is not None
        :param dims2: the sizes for layers in the E-subnet (the *dims paramerter for the `SUB_NET` class), not used if `G_model` or `I_model` is not None
        :param dims3: the sizes for layers in the interaction-subnet (the *dims paramerter for the `SUB_NET` class), not used if `G_model` or `I_model` is not None
        :param F_mat: the correlation matrix/F matrix for the proposed method/ARMI model
        :param cr_type: the type of collaborative regularization, 0: collaborative loss, 1: collaborative loss by genes (proposed), 2: ARMI
        """
        super().__init__()
        self.G_name = G_name
        self.E_name = E_name
        self.I_name = I_name
        self.r = r
        self.p = p
        self.q = q
        self.is_linear = is_linear
        self.cr_type = cr_type
        # Initialize the models
        if G_model is None:
            self.G_model = Model_single_modal(r=r, q=q, is_linear=is_linear,
                     G_name = G_name, E_name=E_name,
                     dims1=dims1, dims2=dims2, dims3=dims3)
        else:
            self.G_model = G_model
        if I_model is None:
            self.I_model = Model_single_modal(r=p, q=q, is_linear=is_linear,
                                              G_name=I_name, E_name=E_name,
                                              dims1=dims1, dims2=dims2, dims3=dims3)
        else:
            self.I_model = I_model

        self.F_mat = F_mat   # p*r

    def forward(self, **tra_X_dic):
        """
        :param tra_X_dic: the input dictionary, including the G, E and I variables,
                          the G varaibles are in the key `G_name` with dimension [n, r],
                          the E variables are in the key `E_name` with dimension [n, q],
                          and the I variables are in the key `I_name` with dimension [n, p]
        :return: pred_G: the predicted value for the G--E interaction model with dimension [n, 1]
                 pred_G_sep: the predicted value for each G main effect and its corresponding interaction terms with dimension [n, r]
                 pred_I: the predicted value for the I--E interaction model with dimension [n, 1]
                 pred_I_sep: the predicted value for each I main effect and its corresponding interaction terms with dimension [n, p]
        """
        pred_G, pred_G_sep = self.G_model(**tra_X_dic)
        pred_I, pred_I_sep = self.I_model(**tra_X_dic)
        return pred_G, pred_G_sep, pred_I, pred_I_sep

    @property
    def coef(self):
        """
        :return: the coefficients (Significance of variables) for the G--E interaction model
        """
        return self.G_model.coef

    @property
    def evaluate_test(self):
        return self.G_model.evaluate_test

    @property
    def evaluate_coef(self):
        return self.G_model.evaluate_coef

    def loss_func(self, out, Y, N_, gamma=0):
        """
        Loss function for the network
        :param out: the output of the network (`forward` function)
        :param Y: the observed survival time
        :param N_: the observed event indicator
        :param gamma: the parameter for the collaborative regularization (lambda_3 in paper)
        :return: the loss value (partial likelihood + collaborative regularization terms)
        """
        pred_G, pred_G_sep, pred_I, pred_I_sep = out
        n = pred_G.shape[0]
        pl =  CoxPHLoss("breslow")(pred_G, torch.stack([Y, N_], dim=1)) + \
               CoxPHLoss("breslow")(pred_I, torch.stack([Y, N_], dim=1))
        if self.cr_type == 0:    # collaborative loss
            cr = gamma * torch.norm(pred_G - pred_I)**2 / n
        elif self.cr_type == 1:  # collaborative loss by genes
            diff = pred_G_sep.reshape(-1, 1, self.r) - pred_I_sep.reshape(-1, self.p, 1)
            cr = gamma * (self.F_mat * diff ** 2).sum() / (n*self.r)
        elif self.cr_type == 2:  # ARMI
            coef_G_mat = self.G_model.coef[:(self.r*(self.q+1))].reshape(self.q+1, self.r) # (q+1)*r
            coef_I_mat = self.I_model.coef[:(self.p*(self.q+1))].reshape(self.q+1, self.p) # (q+q)*p
            diff = coef_G_mat @ self.F_mat.T - coef_I_mat
            cr = gamma * (diff**2).sum() #/ (self.r*(self.q+1))
        else:
            cr = 0
        return pl + cr

    def metric_func(self, log_h, Y, N_):
        """
        C-index
        :param log_h: the predicted log hazard
        :param Y: the observed survival time
        :param N_: the observed event indicator
        :return: the concordance index
        """
        return concordance_index(log_h, Y, N_)

    def prox(self, penalty, **kwargs):
        """
        Proximal operator for the network
        :param penalty: the penalty type
        :param kwargs: the penalty parameters
        """
        lam1_1 = kwargs.get("lam1_1", None)
        lam2_1 = kwargs.get("lam2_1", None)
        lam1_2 = kwargs.get("lam1_2", None)
        lam2_2 = kwargs.get("lam2_2", None)
        coef_G_prox = prox_GE_all(self.G_model.coef, r=self.r, q=self.q,
                                      lam1=lam1_1, lam2=lam2_1,
                                      penalty=penalty
                                     )
        self.G_model.coef.data = coef_G_prox.data

        coef_I_prox = prox_GE_all(self.I_model.coef, r=self.p, q=self.q,
                                  lam1=lam1_2, lam2=lam2_2,
                                  penalty=penalty
                                  )
        self.I_model.coef.data = coef_I_prox.data


class Trainer:
    """
    Trainer for the network
    """
    def __init__(self, model, penalty="our_mcp"):
        """
        :param model: the network model
        :param penalty: the penalty type
        """
        self.model = model
        self.penalty = penalty
        self.best_tunings = None
        self.best_metric = -np.inf
        self.path = dict()

    def train(self, tra_X_dic, tra_Y_dic,
              kwarg_prox={}, kwarg_loss={}, lr=0.5, maxit=50, tol=5e-4, init_state_dict=None,
              val_X_dic=None, val_Y_dic=None,
              prox_it=5, min_it=6, eval_it=5, early_stop_round = 10, use_refit=False
              ):
        """
        Proximal gradient algorithm to train the network
        :param tra_X_dic: the input dictionary for the training set
        :param tra_Y_dic: the output dictionary for the training set
        :param kwarg_prox: the parameters for the `prox` function in the model
        :param kwarg_loss: the parameters for the `loss_func` function in the model
        :param lr: the learning rate
        :param maxit: the maximum number of iterations
        :param tol: the tolerance for the convergence
        :param init_state_dict: the initial state for the network
        :param val_X_dic: the input dictionary for the validation set
        :param val_Y_dic: the output dictionary for the validation set
        :param prox_it: the number of iterations for starting the proximal operator in iterations
        :param min_it: the minimum number of iterations
        :param eval_it: the evaluation interval
        :param early_stop_round: the number of rounds for early stopping
        :param use_refit: whether to use refit (for refit, no prox operator is used)
        :return: the best hyperparameters and the best metric value
        """
        # Proximal gradient algorithm
        # model
        model = self.model
        if init_state_dict:
            model.load_state_dict(init_state_dict)
        # optimizer
        param_optimizer = list(model.named_parameters())
        optimizer_grouped_parameters = [
            {'params': [p for name_, p in param_optimizer if "coef" not in name_], 'weight_decay': 0.1, "lr": lr},
            {'params': [p for name_, p in param_optimizer if "coef" in name_], 'weight_decay': 0, "lr": lr}
        ]
        optimizer = AdamW(optimizer_grouped_parameters)
        # Train iterations
        val_metrics = []
        best_metric, best_it, best_state = -np.inf, 0, None
        coef_pre = model.coef.detach().clone()
        for i in range(maxit):
            model.train()
            optimizer.zero_grad()
            out = model(**tra_X_dic)
            loss = model.loss_func(out, **tra_Y_dic, **kwarg_loss)
            loss.backward()
            optimizer.step()

            # Conduct proximal operator
            if (not use_refit) and (i >= prox_it):
                model.prox(self.penalty, **kwarg_prox)

            # Evaluate the model on the validation set
            if i>min_it and i%eval_it==0 and val_X_dic is not None:
                metric = self.model.evaluate_test(val_X_dic, val_Y_dic)["metric"]
                val_metrics.append(metric)
                if metric > best_metric:
                    best_metric = metric
                    best_it = i
                    best_state = quick_deepcopy(model.state_dict())
                else:
                    # early stopping if the metric does not increase for `early_stop_round` rounds
                    if i - best_it > early_stop_round:
                        break 

            if i > min_it and ((model.coef - coef_pre) ** 2).sum() <= 20 * tol:
                break

            coef_pre = model.coef.detach().clone()

        if val_X_dic is not None and best_state is not None:
            model.load_state_dict(best_state)
        else:
            best_it = i
        return (kwarg_prox, kwarg_loss, lr, best_it), best_metric

    def train_path(self, tra_X_dic, tra_Y_dic,
                   kwarg_prox_set={}, kwarg_loss_set={}, maxit=30, lrs=[0.1, 0.5], tol=1e-3, init_state_dict=None,
                   val_X_dic={}, val_Y_dic={},
                   metric_name = "bic",
                   prox_it=5, min_it=6, eval_it=5, early_stop_round=10, use_refit=False
                   ):
        """
        Train the network with a set of hyperparameters and selected the best one 
        :param tra_X_dic: the input dictionary for the training set
        :param tra_Y_dic: the output dictionary for the training set
        :param kwarg_prox_set: the set of hyperparameters for the proximal operator
        :param kwarg_loss_set: the set of hyperparameters for the loss function
        :param maxit: the maximum number of iterations
        :param lrs: the set of learning rates
        :param tol: the tolerance for the convergence
        :param init_state_dict: the initial state for the network
        :param val_X_dic: the input dictionary for the validation set
        :param val_Y_dic: the output dictionary for the validation set
        :param metric_name: the metric for selecting the best hyperparameters, including "bic", "aic", "metric"
        :param prox_it: the number of iterations for starting the proximal operator in iterations
        :param min_it: the minimum number of iterations
        :param eval_it: the evaluation interval
        :param early_stop_round: the number of rounds for early stopping
        :param use_refit: whether to use refit (for refit, no prox operator is used)
        """
        best_metric = -np.inf
        best_model = None
        best_tunings = None
        if init_state_dict is None:
            init_state_dict = quick_deepcopy(self.model.state_dict())

        names_prox, cand_prox = zip(*list(kwarg_prox_set.items()))
        names_loss, cand_loss = zip(*list(kwarg_loss_set.items()))

        # Grid search for hyperparameters
        i = 0
        for x, lst_prox in enumerate(product(*cand_prox)):
            for y, lst_loss in enumerate(product(*cand_loss)):
                for z, lr in enumerate(lrs):
                    kwarg_prox = {name: val for name, val in zip(names_prox, lst_prox)}
                    kwarg_loss = {name: val for name, val in zip(names_loss, lst_loss)}

                    tunings, metric = self.train(tra_X_dic, tra_Y_dic,
                               kwarg_prox=kwarg_prox, kwarg_loss=kwarg_loss, lr=lr, maxit=maxit, tol=tol, init_state_dict=init_state_dict,
                               val_X_dic=val_X_dic, val_Y_dic=val_Y_dic,
                               prox_it=prox_it, min_it=min_it, eval_it=eval_it, early_stop_round=early_stop_round,
                                                 use_refit=use_refit
                               )

                    if metric_name.endswith("ic"):
                        metric = -self.model.evaluate_test(tra_X_dic, tra_Y_dic)[metric_name]

                    self.path[(tuple(kwarg_prox.items()), tuple(kwarg_loss.items()))] = metric
                    if i == 0 or (metric > best_metric and (self.model.coef != 0).sum() > 0):
                        best_metric = metric
                        best_model = quick_deepcopy(self.model.state_dict())
                        best_tunings = tunings

                    i += 1

        self.model.load_state_dict(best_model)
        self.best_tunings = best_tunings
        self.best_metric = best_metric

