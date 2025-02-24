import numpy as np
from scipy.optimize import root
import torch

def sigmoid(z):
    return 1/(1 + np.exp(-z))

def elu(a, alpha=1):
    return a*(a>0) + alpha*(np.exp(a)-1)*(a<=0)


def generate_data(n=1000, n_val=200, n_tes=2000, r=100, p=100, q=5,
                  case="linear", H=None, H_type="diagnal",
                  n_main=12, n_interactions=14,
                  censor_rate=0.3,
                  coef=None, rho=0.5, seed=42):
        """
        Generate data for simulation
        :param n: number of training samples
        :param n_val: number of validation samples
        :param n_tes: number of test samples
        :param r: number of G variables
        :param p: number of I variables
        :param q: number of E variables
        :param case: "linear", "nonlinear1", "nonlinear2", "nonlinear3", "nonlinear4"
        :param H: (r*p) matrix, the true relationship between G and I variables 
        :param H_type: "diagnal", "band1", "band2", "milky_way", or "nonlinear"
        :param n_main: number of significant main G effects in true model
        :param n_interactions: number of significant interaction effects in true model
        :param censor_rate: censor rate
        :param coef: true coefficients (indicating the significance of each variable)
        :param rho: the parameter of the AR structure covariance matrix of G variables (and E variables)  
        :param seed: random seed
        """
        # ==== initialize ====
        np.random.seed(seed)
        n_all = n+n_val+n_tes

        if not coef:
            # main G effects
            idxs_main = np.array(range(n_main))
            #idxs_main = np.random.choice(range(min(300, r)), size=n_main, replace=False)
            beta = np.zeros(r)
            beta[idxs_main] = np.random.uniform(0.5, 1, size=n_main)

            # main E effects
            alpha = np.random.uniform(0.5, 1, size=q)

            # Interaction effects
            idxs_inter = np.random.choice(range(n_main*q), size=n_interactions, replace=False)
            rs, cs = idxs_inter%q, idxs_main[idxs_inter//q]
            Gamma = np.zeros((q, r))
            Gamma[rs, cs] = np.random.uniform(0.5, 1, size=n_interactions)

            coef = np.concatenate([beta, Gamma.reshape(-1), alpha])

        # H is (r*p) matrix
        if H is None:
            H = np.diag(np.ones(min(r,p)))
            if r > p:
                H = np.concatenate([H, np.zeros((r - p, p))], axis=0)
            else:
                H = np.concatenate([H, np.zeros((r, p - r))], axis=1)

            if H_type == "diagnal":
                pass
            elif H_type == "band1" or H_type == "nonlinear":
                rs, cs = zip(*[(x, y) for x in range(r) for y in range(p) if abs(x-y) == 1])
                H[rs, cs] = 0.3
            elif H_type == "band2":
                rs, cs = zip(*[(x, y) for x in range(r) for y in range(p) if abs(x - y) == 1])
                H[rs, cs] = 0.5
                rs, cs = zip(*[(x, y) for x in range(r) for y in range(p) if abs(x - y) == 2])
                H[rs, cs] = 0.3
            elif H_type == "milky_way":
                size = int(r*p*0.02)
                rnd = np.random.choice(r*p, size=size, replace=False)
                rs, cs = rnd//p, rnd%p
                rs, cs = zip(*[(x, y) for x, y in zip(rs, cs) if x != y])
                vals = np.random.uniform(0.3, 0.6, len(rs))
                H[rs, cs] = vals
                pass

        # ==== generate covariates ====
        # generate E
        mean_vec = np.array([0] * q)
        cov_matrix = np.array([[0.3 ** (abs(i - j)) for j in range(q)] for i in range(q)])
        E_all = np.random.multivariate_normal(mean_vec, cov_matrix, size=n_all)

        # generate Z
        mean_vec = np.array([0]*r)
        cov_matrix = np.array([[rho**(abs(i-j)) for j in range(r)] for i in range(r)])
        Z_all = np.random.multivariate_normal(mean_vec, cov_matrix, size=n_all)

        # generate X
        U_all = np.random.normal(0, 0.05, size=n_all*p).reshape(n_all, p)
        if H_type.endswith("nonlinear"):
            X_all = np.tanh(Z_all.dot(H)) + U_all
        else:
            X_all = Z_all.dot(H) + U_all

        # ==== generate response ====
        W = np.concatenate([Z_all], axis=1)
        for k in range(q):
            V = (Z_all.T * E_all[:, k]).T
            W = np.concatenate([W, V], axis=1)
        W = np.concatenate([W, E_all], axis=1)

        if case == "linear":
            y = W.dot(coef).reshape(-1)
        elif case == "nonlinear1":
            idxs = np.where(coef != 0)[0]
            idxs_main = np.where(coef[:r] != 0)[0]
            idxs_inter = np.where(coef[r:(r * (q + 1))])[0] + r
            y = np.sin(W[:, idxs_main]).sum(axis=1) + 1.5*np.sin(W[:, idxs_inter]).sum(axis=1) + np.sin(W[:, -q:]).sum(axis=1)
            #y = np.sin(W[:, idxs]).sum(axis=1)
        elif case == "nonlinear2":
            idxs_main = np.where(coef[:r]!=0)[0]
            idxs_inter = np.where(coef[r:(r*(q+1))])[0] + r
            y = 0.6*np.tanh(W[:, idxs_main]).sum(axis=1) + np.tanh(W[:, idxs_inter]).sum(axis=1) + 0.4*sigmoid(W[:, -q:]).sum(axis=1)
            #y = 0.6*np.abs(W[:, idxs]).sum(axis=1)
        elif case == "nonlinear3":
            idxs = np.where(coef != 0)[0]
            #y = 0.5*(W[:, idxs]**2).sum(axis=1)
            y = 0.8*elu(W[:, idxs]).sum(axis=1)
        elif case == "nonlinear4":
            idxs = np.where(coef != 0)[0]
            # idxs_main = np.where(coef[:r] != 0)[0]
            # idxs_inter = np.where(coef[r:] != 0)[0] + r
            # y = np.sin(W[:, idxs_main]).sum(axis=1) + np.sin(W[:, idxs_inter]).sum(axis=1)
            y = 0.6*np.abs(W[:, idxs]).sum(axis=1)

        hazard = np.exp(-2+y)
        T_all = np.random.exponential(1/hazard)
        f = lambda c_rate: 1-np.exp(-1/c_rate*T_all).mean()-censor_rate
        c_rate = root(f, np.exp(-2)).x
        C_all = np.random.exponential(c_rate, size=n_all)
        Y_all = np.min(np.stack([T_all, C_all], axis=1), axis=1)
        N_all = (T_all<=C_all).astype(int)

        tra_data = dict(Z=Z_all[:n, :],
                        X=X_all[:n, :],
                        E=E_all[:n, :],
                        Y=Y_all[:n],
                        N_=N_all[:n]
                        )
        val_data = dict(Z=Z_all[n:(n+n_val), :],
                        X=X_all[n:(n+n_val), :],
                        E=E_all[n:(n+n_val), :],
                        Y=Y_all[n:(n+n_val)],
                        N_=N_all[n:(n+n_val)]
                        )
        tes_data = dict(Z=Z_all[(n + n_val):(n + n_val + n_tes), :],
                        X=X_all[(n + n_val):(n + n_val + n_tes), :],
                        E=E_all[(n + n_val):(n + n_val + n_tes), :],
                        Y=Y_all[(n + n_val):(n + n_val + n_tes)],
                        N_=N_all[(n + n_val):(n + n_val + n_tes)]
                        )
        return tra_data, val_data, tes_data, coef


def data_to_torch(data):
    """
    Convert data (numpy) to torch tensors
    :param data: dictionary of data
    :return: dictionary of torch tensors
    """
    data_X = {}
    data_X["Z"] = torch.from_numpy(data["Z"]).float()
    data_X["X"] = torch.from_numpy(data["X"]).float()
    data_X["E"] = torch.from_numpy(data["E"]).float()
    data_Y = {}
    data_Y["Y"] = torch.from_numpy(data["Y"]).float()
    data_Y["N_"] = torch.from_numpy(data["N_"]).int()
    return data_X, data_Y
