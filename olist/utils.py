import numpy as np


def haversine_distance(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance in kilometers between two points
    on the earth (specified in decimal degrees)
    """
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    km = 6371 * c
    return km


def return_significative_coef(model, threshold=0.05):
    """
    Returns a Series of the model's coefficients (excluding Intercept)
    that are statistically significant (p-value < threshold), sorted.
    """
    coefs = model.params[1:]
    pvalues = model.pvalues[1:]
    significant_coefs = coefs[pvalues < threshold]
    return significant_coefs.sort_values()
