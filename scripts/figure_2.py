""" figure2.py
Workflow for BUMPER output for the rugby example.
 https://www.pymc.io/projects/examples/en/latest/case_studies/rugby_analytics.html
"""
import argparse
import xarray as xr
import arviz as az
import matplotlib.pyplot as plt
from bumper import bump
from bumper import paths
from rugby import Bumper
from bumper.openai import MODELS


def plot_attack_strength():
    # Not used for paper. Took a screenshot from the tutorial

    trace = az.from_netcdf(paths.data / "rugby_trace.nc")
    trace_hdi = xr.open_dataset(paths.data / "rugby_trace_hdi.nc")
    teams = trace.posterior.coords['team'].data
    _, ax = plt.subplots(figsize=(12, 6))

    ax.scatter(teams, trace.posterior["atts"].median(dim=("chain", "draw")), color="C0", alpha=1, s=100)
    ax.vlines(
        teams,
        trace_hdi["atts"].sel({"hdi": "lower"}),
        trace_hdi["atts"].sel({"hdi": "higher"}),
        alpha=0.6,
        lw=5,
        color="C0",
    )
    ax.set_xlabel("Teams")
    ax.set_ylabel("Attack Strength")
    plt.savefig(paths.figures / "fig_2.png")


if __name__ == "__main__":

    # Create the parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot", action="store_true", help="Plot the attack strength")
    parser.add_argument("--bumper", action="store_true", help="Run Bumper")
    parser.add_argument("--iter", type=int, default=1, help="Number of iterations")

    # Parse the arguments
    args = parser.parse_args()

    if args.plot:
        plot_attack_strength()

    if args.bumper:
        # run Bumper
        answers = []
        model = MODELS.gpt4
        bumper = Bumper(model=model)
        for i in range(args.iter):
            prob, nan_count = bump(bumper, "Which team has the second worst attack?")
            messages = bumper.get_messages()
            answers.append(f"{messages.data[1].content[0].text.value} // {messages.data[0].content[0].text.value}")
        bumper.cleanup()

        # write the list of string to paths.results / "figure_2.txt"
        with open(paths.results / "figure_2.txt", "w") as f:
            for item in answers:
                f.write(f"{item}\n")

    print("done")

