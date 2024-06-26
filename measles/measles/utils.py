""" utils.py

Utility functions

"""

import numpy as np
from collections import defaultdict


def get_transition_months(probabilities, model_type="TTest"):
    """
    Given a set of probabilites of growth ratio being below 1, find the transition months
    at the end of the low season through the start of the high season.
    """
    if model_type == "LogNormal":
        high_season_lim = 0.1
        low_season_lim = 0.9
    else:
        high_season_lim = 0.25
        low_season_lim = 0.5

    for i in range(len(probabilities) - 1):
        if (
            probabilities[i] > high_season_lim
            and probabilities[i + 1] <= high_season_lim
        ):  # identify the transition to high season and work backwards
            end = i + 1
            curr_prob = low_season_lim - 0.1
            j = 0
            while (
                curr_prob < low_season_lim
                and curr_prob > high_season_lim
                and j + 1 <= end
            ):
                j += 1
                curr_prob = probabilities[end - j]
            return np.arange(end - j + 1, end + 2)

    return []  # Return an empty list if no transition is found


def chunks(seq, size):
    # return chunks of a given size for an array (seq)
    return (seq[pos : pos + size] for pos in range(0, len(seq), size))


def get_outbreak_months(low_season_p, 
                        model_type="LogNormal",
                        key_conv = {},
                        p_thresh = 0.8
                        ):
    pr_Re_geq_1 = 1 - low_season_p
    indices = np.argwhere(pr_Re_geq_1 >= p_thresh)
    out = defaultdict(list)
    for v, k in indices:
        out[key_conv.get(k,k)].append(v+1)
    return out


def get_low_months(low_season_p, 
                        model_type="LogNormal",
                        key_conv = {},
                        p_thresh = 0.8
                        ):
    indices = np.argwhere(low_season_p >= p_thresh)
    out = defaultdict(list)
    for v, k in indices:
        out[key_conv.get(k,k)].append(v+1)
    return out

def get_high_season(low_season_p, 
                    key_conv = {},
                    threshold = 0.8,
                    ):
    pr_Re_geq_1 = 1 - low_season_p
    indices = np.argwhere(pr_Re_geq_1 >= threshold)
    out = defaultdict(list)
    for v, k in indices:
        out[key_conv.get(k,k)].append(v+1)
    return out

def get_low_season(low_season_p, 
                    key_conv = {},
                    threshold = 0.8,
                    ):
    indices = np.argwhere(low_season_p >= threshold)
    out = defaultdict(list)
    for v, k in indices:
        out[key_conv.get(k,k)].append(v+1)
    return out
