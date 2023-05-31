import torch

def zero_one_normalization(X, lower=None, upper=None):

    if lower is None:
        lower = torch.min(X, axis=0)
    if upper is None:
        upper = torch.max(X, axis=0)

    X_normalized = torch.true_divide((X - lower), (upper - lower))

    return X_normalized, lower, upper


def zero_one_denormalization(X_normalized, lower, upper):
    return lower + (upper - lower) * X_normalized


def zero_mean_unit_var_normalization(X, mean=None, std=None):
    if mean is None:
        mean = torch.mean(X, axis=0)
    if std is None:
        std = torch.std(X, axis=0)

    X_normalized = (X - mean) / std

    return X_normalized, mean, std


def zero_mean_unit_var_denormalization(X_normalized, mean, std):
    return X_normalized * std + mean
