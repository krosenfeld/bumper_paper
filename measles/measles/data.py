import sys
import pandas as pd
import numpy as np


def process_data_case_ratios(
    filename,
    countries_list=["Pakistan", "Chad", "Ethiopia", "Madagascar", "Nigeria"],
    long_return=False,
):
    """
    Process raw WHO case data for the TTest class to use in linear regression/seasonality profiles.
    """
    rawdata = pd.read_csv(filename)
    rawdata = rawdata[
        [
            "Country",
            "Year",
            "January",
            "February",
            "March",
            "April",
            "May",
            "June",
            "July",
            "August",
            "September",
            "October",
            "November",
            "December",
        ]
    ].fillna(0)
    rawdata = rawdata[rawdata["Country"].isin(countries_list)]

    long_format = rawdata.melt(
        id_vars=["Country", "Year"],
        value_vars=[
            "January",
            "February",
            "March",
            "April",
            "May",
            "June",
            "July",
            "August",
            "September",
            "October",
            "November",
            "December",
        ],
        var_name="Month",
        value_name="Cases",
    )

    long_format["time"] = pd.to_datetime(
        long_format["Month"].astype("str")
        + " "
        + long_format["Year"].astype("str")
        + " 15",
        format="%B %Y %d",
    )

    if sys.version_info >= (3, 8):
        long_format["cases_next_month"] = (
            long_format.sort_values("time")
            .groupby("Country")["Cases"]
            .apply(lambda x: x.shift(-1)).reset_index(level='Country', drop=True)
        )
    else:
        long_format["cases_next_month"] = (
            long_format.sort_values("time")
            .groupby("Country")["Cases"]
            .apply(lambda x: x.shift(-1))
        )    

    long_format["case_ratio_adjusted"] = np.sqrt(
        (long_format["cases_next_month"] + 1) / (long_format["Cases"] + 1)
    )
    long_format.dropna(subset=["case_ratio_adjusted"], inplace=True)
    long_format.drop(["Month", "Year"], axis=1, inplace=True)

    if long_return:
        return long_format
    else:
        tr_data = long_format.pivot(
            index="time", columns="Country", values="case_ratio_adjusted"
        )
        return tr_data


def process_sia_doses(filename):
    
    ## Get the SIA dose data
    sia_cal = pd.read_csv(filename,
                            header=1,
                            usecols=["Country","Start date","End date",
                                    "Target population","Reached population"],
                            )
    # sia_cal["time"] = pd.to_datetime(sia_cal["Start date"]\
    #                     .str.replace("-","/15/20"),
    #                 errors="coerce")
    sia_cal["time"] = sia_cal["Start date"].str.replace("-", "/15/20").map(lambda s: pd.to_datetime(s, errors="coerce"))
    sia_cal["doses"] = pd.to_numeric(sia_cal["Reached population"]\
                                    .str.replace(" ",""))
    sia_cal["doses"] = sia_cal["doses"].fillna(
        pd.to_numeric(sia_cal["Target population"]\
                                    .str.replace(" ","")))
    
    return sia_cal
