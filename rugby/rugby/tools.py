import json
import numpy as np
import arviz as az
import xarray as xr

from bumper import paths

__all__ = ["tool_specs", "tool_functions"]

tool_specs = []
tool_functions = {}

def zip_it(x, ndigits=2):
    xarray_like = x.to_dict()
    team_names = xarray_like['coords']['team']['data']
    team_data = xarray_like['data']
    d =  dict(zip(team_names, team_data))
    for k,v in d.items():
        d[k] = float(np.round(v,ndigits))
    return d

def get_attack_stats():
    """ Return the attack statistics for the rugby dataset providing the mean and minimum width Bayesian credible intervals (lower and upper bounds). Higher is better. """
    trace_hdi = xr.open_dataset(paths.data / "rugby_trace_hdi.nc")
    trace = az.from_netcdf(paths.data / "rugby_trace.nc")

    m = zip_it(trace.posterior["atts"].median(dim=("chain", "draw")))
    lb = zip_it(trace_hdi["atts"].sel({"hdi": "lower"})) # lower bounds
    ub  = zip_it(trace_hdi["atts"].sel({"hdi": "higher"})) # upper bounds

    d = {}
    for team in lb.keys():
        d[team] = m[team]
        # d[team] = {"median":m[team], "lower_bound": lb[team], "higher_bound": ub[team]}
    print(d)
    return json.dumps(d)


tool_functions["get_attack_stats"] = get_attack_stats
tool_specs.append(
    {
        "type": "function",
        "function": {
            "name": "get_attack_stats",
            "description": get_attack_stats.__doc__,
            "parameters": {},
        },
    }
)


def get_defense_stats():
    """ Return the defense statistics for the rugby dataset providing the mean and minimum width Bayesian credible intervals (lower and upper bounds). Lower is better. """
    trace_hdi = xr.open_dataset(paths.data / "rugby_trace_hdi.nc")
    trace = az.from_netcdf(paths.data / "rugby_trace.nc")

    m = zip_it(trace.posterior["defs"].median(dim=("chain", "draw")))
    lb = zip_it(trace_hdi["defs"].sel({"hdi": "lower"})) # lower bounds
    ub  = zip_it(trace_hdi["defs"].sel({"hdi": "higher"})) # upper bounds

    d = {}
    for team in lb.keys():
        d[team] = {"median":m[team], "lower_bound": lb[team], "higher_bound": ub[team]}
    return json.dumps(d)


tool_functions["get_defense_stats"] = get_defense_stats
tool_specs.append(
    {
        "type": "function",
        "function": {
            "name": "get_defense_stats",
            "description": get_defense_stats.__doc__,
            "parameters": {},
        },
    }
)


if __name__ == "__main__":
    s = get_defense_stats()
    print(s)

    s = get_defense_stats()
    print(s)    