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
    # cols2 = [[utils.c8c['red'], utils.c8c['orange']],[utils.c8c['violet'], utils.c8c['black']]]
    cols2 = [utils.c8c['violet'], utils.c8c['black']]
    models = [MODELS.gpt3, MODELS.gpt4, MODELS.gpt4o]
    bins = np.linspace(-1,1,int(200/25)+1)

    fig, axes = plt.subplots(4, 3, figsize=(8, 4*1.6), sharey=True)
    kwargs = dict(bins=bins,  x='prob', legend=True, hue_order=[True, False], stat='probability', hue='e_flag')
    for ie in range(2):
        for iq, Q in enumerate(Qs):
            for ix, model in enumerate(models):
                    iy = axes.shape[0] - iq - 1 - (ie * 2)
                    ax = axes[iy, ix]
                    with open(paths.results / f"stability_{model}_Q{iq}.json", "r") as f:
                        res = json.load(f)
                    res = pd.DataFrame(res)
                    
                    if ie == 1:
                        res = res[np.logical_not(res['e_flag'])]

                        with open(paths.results / f"stability_{model}_Q{iq}_element.json", "r") as f:
                            res1 = json.load(f)
                        res1 = pd.DataFrame(res1)
                        res1 = res1[res1['e_flag']]
                        if len(res1) > len(res): # use the sample number of samples for histplot normalization
                            res1 = res1.sample(n=len(res), random_state=42)
                        res = pd.concat((res, res1), ignore_index=True)

                    res['prob'] /= 100

                    sns.histplot(res,ax=ax,palette={False:cols[iq], True:cols2[ie]}, **kwargs)

                    # keep the same hue color as in previous figure
                    for bar, hue in zip(ax.patches, ax.get_legend().texts):
                        if hue.get_text() == 'False':
                            fc = bar.get_facecolor()
                    for bar in ax.patches:
                        if bar.get_facecolor() == fc:                    
                            bar.set_facecolor(fc[:-1] + (0.75,))
                    # disable the legend
                    ax.get_legend().remove()

                    utils.axes_setup(ax, left_visible=False)
                    utils.set_xaxis(ax)
                    ax.set_yticks([])
                    ax.set_xlabel("")
                    ax.set_ylabel("")          
                    if iy != (axes.shape[0] - 1):
                        ax.set_xticklabels([])
                        # ax.set_title(model)
                    if ix == 0:
                        ax.text(-0.1, 0.6, utils.insert_newline_midway(f"{Q}"), ha='left', va='center', transform=ax.transAxes, fontsize=9)
                    if ix == 1 and iy == (axes.shape[0] - 1):
                        ax.set_xlabel("guideline check")
                    if iy == 0:
                        ax.text(0.5,0.9,model, ha='center', va='bottom', transform=ax.transAxes, fontsize=8)
                    if iq == 1 and ix == 0:
                        ax.text(-0.1,0.87,['b)','a)'][ie], ha='left', va='center', transform=ax.transAxes, fontsize=12)

    # add an axes that covers the entire figure
    ax = fig.add_subplot(111, frame_on=False)
    # make sure the axis does not have any color in the background
    ax.patch.set_visible(False)
    ax.set_yticks([]); ax.set_xticks([])
    ax.plot([-0.03,1.03], 2*[0.49], color="k", transform=ax.transAxes, clip_on=False, linewidth=0.7)

    fig.tight_layout()
    plt.savefig(paths.figures / "fig_5.png")


if __name__ == "__main__":
    make_figure()