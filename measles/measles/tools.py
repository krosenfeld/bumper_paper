import json
import pycountry
import numpy as np
import pandas as pd
from bumper import paths
from openai import OpenAI
from .data import process_data_case_ratios, process_sia_doses
from .LogNormal import LogNormal
from . import utils

START_DATE = "2014-01-01"
END_DATE = "2024-02-01"

__all__ = ["tool_specs", "tool_functions"]

tool_specs = []
tool_functions = {}

######################## Seasonality ########################

def get_sia_months(location):
    """ Get months when measles supplementary immunization activities (SIAs) are recommended to occur for a given country """
    # from iso3 get country name
    location = pycountry.countries.get(alpha_3=location).name
    print(location)
    countries_list = [location]

    res = {}

    # Process raw case data
    data = process_data_case_ratios(
        paths.data / "measlescasesbycountrybymonth_Mar2024.csv",
        long_return=True,
        countries_list=countries_list,
    )
    data = data.loc[data["time"] <= END_DATE]

    # Fit the seasonality model
    logt = LogNormal(data)

    ## Get the transition months
    high_seasons = utils.get_high_season(logt.p_low,
                                   key_conv=logt.i_to_c)
    low_seasons = utils.get_low_season(logt.p_low,
                                 key_conv=logt.i_to_c)
    
    ## Get post-low-season transition months
    ls = low_seasons[location]
    hs = high_seasons[location]
    if len(ls) != 0 and len(hs) != 0: ## seasonal country
        low_end = max(low_seasons[location])
        i = low_end
        sia_months = []
        while i%12+12*(i%12 == 0) not in hs:
            m = i%12 + 12*(i%12 == 0)
            sia_months.append(m)
            i += 1
    elif len(ls) != 0: ## Just a low season
        sia_months = list(set(range(1,13)) - set(ls))
    elif len(hs) != 0: ## Just a high season
        sia_months = list(set(range(1,13)) - set(hs))

    res["sia_months"] = ', '.join([str(month) for month in sia_months])

    return json.dumps(res)


tool_functions["get_sia_months"] = get_sia_months
tool_specs.append(
    {
        "type": "function",
        "function": {
            "name": "get_sia_months",
            "description": get_sia_months.__doc__,
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "A real country in iso3 format, e.g., TCD for Chad",
                    },
                    "unit": {"type": "string"},
                },
                "required": ["location"],
            },
        },
    }
)

def get_high_months(location):
    """ Get months when measles transmission is high (high season) """

    # from iso3 get country name
    location = pycountry.countries.get(alpha_3=location).name
    countries_list = [location]

    res = {}

    # Process raw case data
    data = process_data_case_ratios(
        paths.data / "measlescasesbycountrybymonth_Mar2024.csv",
        long_return=True,
        countries_list=countries_list,
    )
    data = data.loc[data["time"] <= END_DATE]

    # Fit the seasonality model
    logt = LogNormal(data)
    
    # Compute high-season months
    outbreak_months = utils.get_outbreak_months(logt.p_low,
                        key_conv=logt.i_to_c)[countries_list[0]]
    res["high_months"] = ', '.join([str(month) for month in outbreak_months])

    return json.dumps(res)

tool_functions["get_high_months"] = get_high_months
tool_specs.append(
    {
        "type": "function",
        "function": {
            "name": "get_high_months",
            "description": get_high_months.__doc__,
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "A real country in iso3 format, e.g., TCD for Chad",
                    },
                    "unit": {"type": "string"},
                },
                "required": ["location"],
            },
        },
    }
)

def get_low_months(location):
    """ Get months when measles transmission is low (low-season) """

    # from iso3 get country name
    location = pycountry.countries.get(alpha_3=location).name
    countries_list = [location]

    res = {}

    # Process raw case data
    data = process_data_case_ratios(
        paths.data / "measlescasesbycountrybymonth_Mar2024.csv",
        long_return=True,
        countries_list=countries_list,
    )
    data = data.loc[data["time"] <= END_DATE]

    # Fit the seasonality model
    logt = LogNormal(data)
    
    # Compute low-season months
    low_months = utils.get_low_months(logt.p_low,
                        key_conv=logt.i_to_c)[countries_list[0]]
    res["low_months"] = ', '.join([str(month) for month in low_months])

    return json.dumps(res)

tool_functions["get_low_months"] = get_low_months
tool_specs.append(
    {
        "type": "function",
        "function": {
            "name": "get_low_months",
            "description": get_low_months.__doc__,
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The country in iso3 format, e.g., TCD for Chad",
                    },
                    "unit": {"type": "string"},
                },
                "required": ["location"],
            },
        },
    }
)

######################## Susceptibility ########################

def get_susceptibility_forecast(location):
    """ Estimate future relative susceptibility, a proxy for risk of measles outbreaks """

    # RiskAsssessment.py

    # from iso3 get country name
    location = pycountry.countries.get(alpha_3=location).name
    countries_list = [location]

    # Process raw case data
    data = process_data_case_ratios(
        paths.data / "measlescasesbycountrybymonth_Mar2024.csv",
        long_return=True,
        countries_list=countries_list,
    )
    data = data.loc[data["time"] <= END_DATE]

    # Fit the seasonality model
    logt = LogNormal(data)
    
    # Run the susceptibility reconstruction
    sia_cal = process_sia_doses(paths.data / "Summary_MR_SIA.csv")
    cal = sia_cal.loc[sia_cal["Country"] == countries_list[0]]
    logt.run_reconstruction(cal)

    # generate the forecast
    (Zt_extrap, _) = logt.extrapolate()

    info = "When the relative susceptibility is greater than 1, the risk of a measles outbreak is higher than average."
    return json.dumps({"info":info, "susceptibility_forecast": Zt_extrap.to_string()})


tool_functions["get_susceptibility_forecast"] = get_susceptibility_forecast
tool_specs.append(
    {
        "type": "function",
        "function": {
            "name": "get_susceptibility_forecast",
            "description": get_susceptibility_forecast.__doc__,
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "A real country in iso3 format, e.g., TCD for Chad",
                    },
                    "unit": {"type": "string"},
                },
                "required": ["location"],
            },
        },
    }
)


if __name__ == "__main__":
    get_susceptibility_forecast('NGA')
    print("done")