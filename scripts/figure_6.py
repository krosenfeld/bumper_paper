import argparse
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.cm import ScalarMappable
from matplotlib.colors import ListedColormap

import distance
import umap
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

from bumper import paths
from bumper.openai import MODELS
from bumper import utils

plt.rcParams.update({"figure.dpi": 350})
plt.rcParams.update({"font.size": 10})


def load_data(e_flag=False):
    # load all the data
    df = pd.DataFrame()
    for iq in [0]:
        for model in [MODELS.gpt4]:
            tmp = pd.read_json(
                paths.results / f"state_space_{model}_Q{iq}_embeddings.json"
            )
            tmp["model"] = model
            df = pd.concat((df, tmp), ignore_index=True)
    df["prob"] /= 100

    # select the example we are interested in
    if not e_flag:
        print("selecting e_flag = False")
        df = df[np.logical_not(df["e_flag"])]
    elif e_flag:
        print("selecting e_flag = True")
        df = df[df["e_flag"]]
    else:
        print("selecting all e_flags")

    data = np.array(df["embedding"].to_list())
    scaler = StandardScaler().fit(data)

    scaled_data = scaler.transform(data)
    return df, scaled_data


def make_embedding():
    _, scaled_data = load_data()

    # reletavely expensive
    reducer = umap.UMAP(n_components=2, n_neighbors=18, random_state=12)
    mapper = reducer.fit(scaled_data)
    pickle.dump(mapper, open(paths.results / "mapper.pkl", "wb"))


def make_plot():
    df, scaled_data = load_data()
    mapper = pickle.load(open(paths.results / "mapper.pkl", "rb"))
    embedding = mapper.transform(scaled_data)

    pdf = df.copy()
    pdf["feature0"] = embedding[:, 0]
    pdf["feature1"] = embedding[:, 1]

    # k-means clustering on embedding
    kmeans = KMeans(n_clusters=5, random_state=15).fit(embedding)

    # get the cluster centers
    centers = kmeans.cluster_centers_
    labels = kmeans.labels_
    pdf["label"] = labels

    # make figure to get ranges
    plt.figure()
    ax = plt.gca()
    cm = ax.scatter(*pdf[["feature0", "feature1"]].to_numpy().T)
    xra = ax.get_xlim()
    yra = ax.get_ylim()

    # figures
    cmap = "coolwarm_r"
    # compute jacard similarity for each cluster
    jaccard = {}
    for group_name, group in pdf.groupby("label"):
        j = []
        for i in range(20):
            # sample two indices of group without replacement
            idx = np.random.choice(group.index, 2, replace=False)
            j.append(
                distance.jaccard(
                    group.loc[idx[0]]["answer"], group.loc[idx[1]]["answer"]
                )
            )
        jaccard[group_name] = j

    # plot the cluster centers
    h = 0.2  # point in the mesh [x_min, x_max]x[y_min, y_max].
    xx, yy = np.meshgrid(np.arange(*xra, h), np.arange(*yra, h))
    # Obtain labels for each point in mesh. Use last trained model.
    X = np.c_[xx.ravel(), yy.ravel()].astype(embedding.dtype)
    Z = kmeans.predict(X)
    Z = Z.reshape(xx.shape)

    # Create a scalar mappable object with the new colormap
    norm = plt.Normalize(vmin=-1, vmax=1)  # Adjust normalization as needed
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    fig, axes = plt.subplots(1, 1, figsize=(7, 4))
    axes = [axes]

    # create a colormap that goes from 0 to 0.8 in the gist_gray_r colormap
    gist_gray_r = plt.get_cmap("gist_gray_r")
    newcmp = ListedColormap(gist_gray_r(np.linspace(0, 0.75, 256)))

    # UMAP embedding
    ax = axes[0]
    ax.imshow(
        Z,
        interpolation="nearest",
        extent=(xx.min(), xx.max(), yy.min(), yy.max()),
        cmap=newcmp,
        aspect="auto",
        origin="lower",
    )
    cm = ax.scatter(
        *pdf[["feature0", "feature1"]].to_numpy().T,
        s=75,
        c=pdf["prob"],
        cmap=cmap,
        alpha=0.3,
        vmin=-1,
        vmax=1,
        edgecolors="None",
    )
    cb = plt.colorbar(sm, ax=ax)
    ticks = [-1, -0.5, 0, 0.5, 1]
    cb.set_ticks(ticks)  # Set the positions of the ticks
    tickl = [str(t) for t in np.abs(ticks)]
    tickl[0] = tickl[0] + "/fail"
    tickl[-1] = tickl[-1] + "/pass"
    cb.set_ticklabels(tickl, fontsize=8)  # Set the tick labels
    cb.set_label("guideline check", rotation=270, labelpad=5)
    # cbar.set_ticklabels(tick_labels)  # Set custom tick labels

    for label, center in enumerate(centers):
        # plt.text(*center, f' {label}', fontsize=18, color='k', ha='left', va='top')
        v = jaccard[label]
        kwargs = {"fontsize": 10, "ha": "center", "va": "top", "color": "k"}
        if np.mean(v) > 0:
            ax.text(
                center[0],
                center[1] - 1,
                f"$C_{label}$:{np.mean(v):.2f} +/- {np.std(v):.2f}",
                **kwargs,
            )
        else:
            ax.text(center[0], center[1] - 1, f"$C_{label}$:{np.mean(v):.0f}", **kwargs)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    fig.tight_layout()
    plt.savefig(paths.figures / "fig_6.png")

    tinydf = pdf[["label", "prob", "answer"]].sort_values("label")
    tinydf.to_csv(paths.results / "clustered.csv", index=False)


def write_table():
    # load paths.data / clusterd.csv
    df = pd.read_csv(paths.results / "clustered.csv")

    TEMPLATE = "\\item ($S={p:0.2f}$)| {a}\n"
    with open(paths.results / "clustered.tab", "w") as f:
        for label, group in df.groupby("label"):
            f.write("\\textbf{{cluster}}:$C_{label}$\n".format(label=label))
            f.write("\\begin{itemize}\n")
            idx = group.sample(5)
            for i in idx.index:
                row = group.loc[i]
                f.write(TEMPLATE.format(p=row["prob"], a=row["answer"]))
            f.write("\\end{itemize}\n")


if __name__ == "__main__":
    # Create the parser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-e", "--embed", action="store_true", help="Make UMAP embedding"
    )
    parser.add_argument("-p", "--plot", action="store_true", help="Make figure")
    parser.add_argument("-w", "--write", action="store_true", help="Write table")

    # Parse the arguments
    args = parser.parse_args()

    if args.embed:
        make_embedding()

    if args.plot:
        make_plot()

    if args.write:
        write_table()
