""" stability_experiments.py

Experiments for figure 4 and figure 5
"""

import json
import argparse
import datetime
from bumper import bump
from bumper import paths
from bumper.openai import MODELS
from bumper.judges import WholeJudge
from measles import Bumper

Qs =  ["When should Cameroon plan SIAs?", "When should the next SIA in Cameroon be planned?"]


def run_experiment_base(n_per_query=25, n_per_answer=3):
    """ Base experiment, check against the whole guideline """
    for iq, Q in enumerate(Qs):
        models = [MODELS.gpt3, MODELS.gpt4, MODELS.gpt4o]
        for model in models:
            res = {'e_flag':[], 'prob':[], 'completion_tokens':[], 'prompt_tokens':[], 'answer':[]}
            for e_flag in [True, False]:
                asst = Bumper(model=model, explain=e_flag, verbose=True)
                asst.judge = WholeJudge(model=asst.model, sop_file=asst.sop_file, explain=asst.explain)
                for i in range(n_per_query):
                    try:
                        probs, nan_count = bump(asst, Q, n=n_per_answer)
                        n = len(probs)
                        res['prob'] += list(probs)
                        res['e_flag'] += [e_flag]*n
                        res['completion_tokens'] += [asst.completion_tokens + asst.judge.completion_tokens]*n
                        res['prompt_tokens'] += [asst.prompt_tokens + asst.judge.prompt_tokens]*n
                        res['answer'] += [asst.get_messages().data[1].content[0].text.value]*n
                    except Exception as e:
                        print(f"Error: {e}")
                asst.summarize_tokens()
                asst.cleanup()

            # write res to ap.paths.results / "compare_sops.json"
            with open(paths.results / f"stability_{model}_Q{iq}.json", "w") as f:
                json.dump(res, f, indent=4)

def run_experiment_element(n_per_query=25, n_per_answer=3):
    """ Experiment checking by element in the guideline """
    for iq, Q in enumerate(Qs):
        models = [MODELS.gpt3, MODELS.gpt4, MODELS.gpt4o]
        for model in models:
            res = {'e_flag':[], 'prob':[], 'completion_tokens':[], 'prompt_tokens':[], 'answer':[]}
            for e_flag in [True]:
                asst = Bumper(model=model, explain=e_flag, verbose=True)
                asst.judge = WholeJudge(model=asst.model, sop_file=asst.sop_file, explain=asst.explain)
                for i in range(n_per_query):
                    try:
                        probs, nan_count = bump(asst, Q, n=n_per_answer)
                        n = len(probs)
                        res['prob'] += list(probs)
                        res['e_flag'] += [e_flag]*n
                        res['completion_tokens'] += [asst.completion_tokens + asst.judge.completion_tokens]*n
                        res['prompt_tokens'] += [asst.prompt_tokens + asst.judge.prompt_tokens]*n
                        res['answer'] += [asst.get_messages().data[1].content[0].text.value]*n
                    except Exception as e:
                        print(f"Error: {e}")
                asst.summarize_tokens()
                asst.cleanup()

            # write res to ap.paths.results / "compare_sops.json"
            with open(paths.results / f"stability_{model}_Q{iq}_element.json", "w") as f:
                json.dump(res, f, indent=4)

if __name__ == "__main__":

    # Create the parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot", action="store_true", help="Plot the attack strength")
    parser.add_argument("-e", "--experiment", type=str, default="nothing", help="Experiment name")

    # Parse the arguments
    args = parser.parse_args()

    # run experiments
    if args.experiment == "base":
        run_experiment_base(n_per_query=25, n_per_answer=3)
        with open(paths.results / "stability_base_timestamp.txt", "w") as f:
            f.write(datetime.datetime.now().isoformat())
    elif args.experiment == "element":
        run_experiment_element(n_per_query=25, n_per_answer=3)
        with open(paths.results / "stability_element_timestamp.txt", "w") as f:
            f.write(datetime.datetime.now().isoformat())
    else:
        print("No experiment specified")