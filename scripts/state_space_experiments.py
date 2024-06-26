""" state_space_experiments.py

Look at similarity/state space between answers
"""

import json
import datetime
from bumper import bump
from bumper import paths
from bumper.openai import MODELS, EMBEDDINGS, Embedder
from bumper.judges import WholeJudge
from measles import Bumper

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import seaborn as sns

plt.rcParams.update({'figure.dpi': 200})

Qs =  ["When should Cameroon plan SIAs?", "When should the next SIA in Cameroon be planned?"]

def run_experiment_whole(n_per_query=200, n_per_answer=1):
    """ Check against all the criteria """
    # Q = "When should the next SIA in Cameroon be planned?"
    models = [MODELS.gpt4]
    # for iq, Q in enumerate(Qs):
    for iq, Q in enumerate([Qs[0]]):
        for model in models:
            res = {'e_flag':[], 'prob':[], 'completion_tokens':[], 'prompt_tokens':[], 'answer':[], 'explanation':[], 'answer_iter':[], 'fails':[]}
            for e_flag in [True, False]:
                asst = Bumper(model=model, explain=e_flag, verbose=True)
                asst.judge = WholeJudge(model=asst.model, sop_file=asst.sop_file, explain=asst.explain)
                for i in range(n_per_query):
                    try:
                        probs, nan_count = bump(asst, Q, n=n_per_answer)
                        n = len(probs)
                        res['prob'] += list(probs)
                        res['e_flag'] += [e_flag]*n
                        res['fails'] += [int(nan_count)]*n                    
                        res['answer'] += [asst.get_messages().data[1].content[0].text.value]*n
                        res['explanation'] += [asst.judge.last_completion.choices[0].message.content]*n
                        res['completion_tokens'] += [asst.completion_tokens + asst.judge.completion_tokens]*n
                        res['prompt_tokens'] += [asst.prompt_tokens + asst.judge.prompt_tokens]*n
                        res['answer_iter'] += [answer_iter for answer_iter in range(n)]
                    except Exception as e:
                        print(f"Error: {e}")
                asst.summarize_tokens()
                asst.cleanup()

            # write res to paths.results / "compare_sops.json"
            with open(paths.results / f"state_space_{model}_Q{iq}.json", "w") as f:
                json.dump(res, f, indent=4)
        

def calc_experiment_embeddings(prefix="state_space_countries"):
    embedder = Embedder(model=EMBEDDINGS.small)
    df = pd.read_json(paths.results / f"{prefix}.json")
    df['embedding'] = None
    # loop through the rows
    for index, row in df.iterrows():
        emb = embedder.get_embedding(row['answer'])
        df.at[index, 'embedding'] = emb
    df.to_json(paths.results / f"{prefix}_embeddings.json")        


if __name__ == "__main__":
    models = [MODELS.gpt3, MODELS.gpt4, MODELS.gpt4o]
    
    # run_experiment_whole()
    with open(paths.results / "state_space_timestamp.txt", "w") as f:
        f.write(datetime.datetime.now().isoformat())
    # for iq, Q in enumerate(Qs):
    #     for model in models:
    #         calc_experiment_embeddings("state_space_{model}_Q{iq}".format(model=model, iq=iq))


    print("done")