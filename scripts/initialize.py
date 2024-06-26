"""initialize.py

Add assets for the BUMPERs
"""

import os
import argparse
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
import arviz as az
from openai import OpenAI
from bumper import paths


def init_measles():
    client = OpenAI()
    vector_store = client.beta.vector_stores.create(
        name="bumper-measles-evidence-vector-store"
    )

    # Ready the files for upload to OpenAI
    file_paths = [paths.data / "2405.09664v1.pdf"]
    file_streams = [open(path, "rb") for path in file_paths]

    # Use the upload and poll SDK helper to upload the files, add them to the vector store,
    # and poll the status of the file batch for completion.
    file_batch = client.beta.vector_stores.file_batches.upload_and_poll(
        vector_store_id=vector_store.id, files=file_streams
    )

    # You can print the status and the file counts of the batch to see the result of this operation.
    print(file_batch.status)
    print(file_batch.file_counts)

    # Store the vector_store.id in paths.data / "vectore-store-evidence.id"
    print("Writing evidence vector store ID to data/measles-evidence-vector-store.id")
    with open(paths.data / "measles-evidence-vector-store.id", "w") as f:
        f.write(vector_store.id)


def init_rugby():
    # https://www.pymc.io/projects/examples/en/latest/case_studies/rugby_analytics.html

    try:
        df_all = pd.read_csv(paths.data / "rugby.csv", index_col=0)
    except:
        df_all = pd.read_csv(pm.get_data("rugby.csv"), index_col=0)

    home_idx, teams = pd.factorize(df_all["home_team"], sort=True)
    away_idx, _ = pd.factorize(df_all["away_team"], sort=True)
    coords = {"team": teams}

    with pm.Model(coords=coords) as model:
        # constant data
        home_team = pm.ConstantData("home_team", home_idx, dims="match")
        away_team = pm.ConstantData("away_team", away_idx, dims="match")

        # global model parameters
        home = pm.Normal("home", mu=0, sigma=1)
        sd_att = pm.HalfNormal("sd_att", sigma=2)
        sd_def = pm.HalfNormal("sd_def", sigma=2)
        intercept = pm.Normal("intercept", mu=3, sigma=1)

        # team-specific model parameters
        atts_star = pm.Normal("atts_star", mu=0, sigma=sd_att, dims="team")
        defs_star = pm.Normal("defs_star", mu=0, sigma=sd_def, dims="team")

        atts = pm.Deterministic("atts", atts_star - pt.mean(atts_star), dims="team")
        defs = pm.Deterministic("defs", defs_star - pt.mean(defs_star), dims="team")
        home_theta = pt.exp(intercept + home + atts[home_idx] + defs[away_idx])
        away_theta = pt.exp(intercept + atts[away_idx] + defs[home_idx])

        # likelihood of observed data
        home_points = pm.Poisson(
            "home_points",
            mu=home_theta,
            observed=df_all["home_score"],
            dims=("match"),
        )
        away_points = pm.Poisson(
            "away_points",
            mu=away_theta,
            observed=df_all["away_score"],
            dims=("match"),
        )
        # determine number of cores on the machine
        ncores = os.cpu_count()
        trace = pm.sample(1000, tune=1500, cores=ncores)

    trace.to_netcdf(paths.data / "rugby_trace.nc")
    trace_hdi = az.hdi(trace)
    trace_hdi.to_netcdf(paths.data / "rugby_trace_hdi.nc")


if __name__ == "__main__":
    # Create the parser
    parser = argparse.ArgumentParser(description="Initialize bumper_paper assets")
    parser.add_argument("--measles", action="store_true")
    parser.add_argument("--rugby", action="store_true")

    # Parse the arguments
    args = parser.parse_args()

    if args.measles:
        init_measles()
    if args.rugby:
        init_rugby()
