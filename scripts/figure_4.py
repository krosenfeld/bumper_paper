import json
import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from bumper import utils
from bumper import paths
from bumper.openai import MODELS

plt.rcParams.update({'figure.dpi': 350})
plt.rcParams.update({'font.size': 10})

Qs =  ["When should Cameroon plan SIAs?", "When should the next SIA in Cameroon be planned?"]

def make_figure():
    cols = [utils.c8c['blue'], utils.c8c['green']]    
    models = [MODELS.gpt3, MODELS.gpt4, MODELS.gpt4o]
    bins = np.linspace(-1,1,int(200/25)+1)

    fig, axes = plt.subplots(2, 3, figsize=(8, 2*1.5))
    for iq, Q in enumerate(Qs):
       for ix, model in enumerate(models):
            iy = len(Qs) - iq - 1
            ax = axes[iy, ix]
            with open(paths.results / f"stability_{model}_Q{iq}.json", "r") as f:
                res = json.load(f)
            res = pd.DataFrame(res)
            res['prob'] /= 100
            print(res.shape)
            # take only those with no explanation
            res = res[np.logical_not(res['e_flag'])]
            kwargs = dict(bins=bins,  x='prob')
            sns.histplot(res,ax=ax,color=cols[iq],**kwargs)
            utils.axes_setup(ax, left_visible=False)
            utils.set_xaxis(ax)
            ax.set_yticks([])
            ax.set_xlabel("")
            ax.set_ylabel("")          
            if iy == 0:
                ax.set_xticklabels([])
                ax.text(0.5,1.1,model, ha='center', va='bottom', transform=ax.transAxes, fontsize=8)
                # ax.set_title(model)
            if ix == 0:
                ax.text(-0.1, 0.8, utils.insert_newline_midway(f"{Q}"), ha='left', va='center', transform=ax.transAxes, fontsize=9)
            if ix == 1 and iy == 1:
                ax.set_xlabel("guideline check")

    fig.tight_layout()
    plt.savefig(paths.figures / "fig_4.png")

if __name__ == "__main__":
    make_figure()