import sys
sys.path.append("Z:/python_packages")
sys.path.extend(['Z:\\Online_heterogeneous', 'Z:/Online_heterogeneous'])
import torch
from torch import Tensor
import numpy as np
from collections import defaultdict


def soft_thresholding(X, lam):
    """
    Soft thresholding operator for L1 regularization
    :param X: input tensor
    :param lam: regularization parameter
    :return: soft-thresholded tensor
    """
    return X.sign() * (X.abs() > lam) * (X.abs() - lam)


def prox_mcp(X, lam, gamma=3, penalty="mcp"):
    """
    MCP proximal operator (element-wise)
    :param X: input tensor
    :param lam: regularization parameter
    :param gamma: tuning parameter
    """
    if penalty == "mcp":
        return (X.abs() < gamma * lam) * soft_thresholding(X, lam) / (1 - 1 / gamma) + \
               (X.abs() >= gamma * lam) * X
    else:
        return soft_thresholding(X, lam)


def grad_mcp(X, lam, gamma=3):
    """
    Gradient of MCP penalty (element-wise)
    :param X: input tensor
    :param lam: regularization parameter
    :return gradient of MCP penalty
    """
    return X.sign() * (X <= lam * gamma) * (lam - X / gamma)


def grad_gmcp(X, lam, gamma=3):
    """
    Gradient of group MCP penalty (all elements in X are viewed as a group)
    :param X: input tensor
    :param lam: regularization parameter
    :param gamma: tuning parameter
    :return: gradient of group MCP penalty
    """
    l2_norm = (X ** 2).sum().sqrt()
    return X / l2_norm * grad_gmcp(l2_norm, lam, gamma)


def prox_gmcp(X, lam, gamma=3, penalty="mcp"):
    """
    Proximal operator for group MCP penalty (all elements in X are viewed as a group)
    :param X: input tensor
    :param lam: regularization parameter
    :param gamma: tuning parameter
    :return: proximal operator for group MCP penalty
    """
    l2_norm = (X ** 2).sum().sqrt()
    if penalty == "mcp":
        if l2_norm <= lam:
            return torch.zeros(X.shape)
        elif l2_norm >= lam*gamma:
            return X
        else:
            X_ = X.clone().detach()
            X_pre = X_.clone().detach()
            for i in range(20):
                l2_norm = (X_ ** 2).sum().sqrt()
                X_ = X / (1 + grad_mcp(l2_norm, lam, gamma) / l2_norm)
                if ((X_ - X_pre) ** 2).mean() <= 1e-10:
                    break
                X_pre = X_.clone()
            return X_
    else:
        if l2_norm <= lam:
            return torch.zeros(X.shape)
        else:
            return X * (1-lam/l2_norm)


def prox_GE(x_, lam1=0.01, lam2=0.01, penalty="our_mcp"):
    """
    Proximal operator for one G main effect and q interaction effects
    :param x_: input tensor with dim (q+1), the first element is the G effect, the rest elements are the corresponding interaction effects
    :param lam1: regularization parameter for the first element in x_
    :param lam2: regularization parameter for the rest elements in x_
    :param penalty: penalty type, including "mcp", "gmcp", "our_mcp"
    :return: proximal results for GE (same dimension as x_)
    """
    if penalty.lower() == "mcp":
        return prox_mcp(x_, lam1)
    elif penalty.lower() == "gmcp":
        return prox_gmcp(x_, lam1, 3, "mcp")
    elif penalty.lower() == "our_mcp":
        x = x_.detach().clone()
        x0 = x_.detach().clone()
        q = x.shape[0] - 1
        val = torch.norm(torch.cat([x[[0]], soft_thresholding(x[1:], lam2)]))
        lam1 = np.sqrt(q+1) * lam1
        if val <= lam1:
            return torch.zeros(q+1)
        ind0 = torch.where(x[1:].abs() <= lam2)[0] + 1
        ind1 = torch.where(x[1:].abs() > lam2)[0] + 1
        x[ind0] = 0
        x_pre = x.detach().clone()
        for _ in range(100):
            norm_ = torch.norm(torch.cat([x[[0]], x[ind1]]))
            den = 1 + grad_mcp(norm_, lam1, 3) / norm_
            x[0] = x0[0] / den
            x[ind1] = (x0[ind1] - grad_mcp(x[ind1], lam2, 3)) / den
            if torch.norm(x[ind1] - x_pre[ind1]) < 1e-3:
                break
            x_pre = x.detach().clone()
        return x
    else:
        return soft_thresholding(x_, lam1)


def prox_GE_all(coef_, r=100, q=4, lam1=0.01, lam2=None, penalty="our_mcp"):
    """
    Proximal operator for all G main effects and interaction effects
    :param coef_: input tensor with dim (r*(q+1)), the first column contain all the G main effects, the rest elements are the corresponding interaction effects
    :param r: number of G variables
    :param q: number of E variables
    :param lam1: first regularization parameter 
    :param lam2: second regularization parameter (optional)
    :param penalty: penalty type, including "mcp", "gmcp", "our_mcp"
    :return: proximal results for all GE (same dimension as
    """
    if lam2 is None:
        lam2 = lam1
    coef = coef_.detach().clone()
    for j in range(r):
        ind = [j] + [k*r+j for k in range(1, q+1)]
        coef[ind] = prox_GE(coef[ind], lam1, lam2, penalty)
    return coef


def prox_zero(coef_, zero_idxs):
    """
    Proximal operator for zero coefficients
    :param coef_: input tensor  
    :param zero_idxs: indices of zero coefficients
    :return: proximal results for zero coefficients
    """
    coef = coef_.detach.clone()
    coef[zero_idxs] = 0
    return coef

