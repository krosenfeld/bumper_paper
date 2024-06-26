import warnings

## Standard imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

## For the log normal cdf, quantiles
from scipy.special import erf, erfinv

from collections import defaultdict

from measles import (LogNormal, process_data_case_ratios)
from bumper import paths

## Plot environment
plt.rcParams["font.size"] = 22.0
# plt.rcParams["font.family"] = "serif"
# plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["xtick.labelsize"] = "medium"
plt.rcParams["ytick.labelsize"] = "medium"
plt.rcParams["legend.fontsize"] = "medium"
plt.rcParams["axes.linewidth"] = 1.0
plt.rcParams["axes.formatter.use_mathtext"] = True
plt.rcParams["mathtext.fontset"] = "cm"
c8c = {
    "red": "#ED0A3F",
    "orange": "#FF8833",
    "yellow": "#FBE870",
    "green": "#01A638",
    "blue": "#0066FF",
    "violet": "#803790",
    "brown": "#AF593E",
    "black": "#000000",
}

def get_outbreak_months(low_season_p, 
                        model_type="LogNormal",
                        key_conv = {},
                        ):
    pr_Re_geq_1 = 1 - low_season_p
    indices = np.argwhere(pr_Re_geq_1 >= 0.8)
    out = defaultdict(list)
    for v, k in indices:
        out[key_conv.get(k,k)].append(v+1)
    return out

def get_low_months(low_season_p, 
                        model_type="LogNormal",
                        key_conv = {},
                        ):
    indices = np.argwhere(low_season_p >= 0.8)
    out = defaultdict(list)
    for v, k in indices:
        out[key_conv.get(k,k)].append(v+1)
    return out

if __name__ == "__main__":

    ## Set up
    countries_list = ["Chad"]
    start_date = "2014-01-01"
    end_date = "2024-02-01"
    extrap_months = 24

    # Process raw case data
    data = process_data_case_ratios(
        paths.data / "measlescasesbycountrybymonth_Mar2024.csv",
        long_return=True,
        countries_list=countries_list,
    )
    data = data.loc[data["time"] <= end_date]

    ## Get the SIA dose data
    sia_cal = pd.read_csv(paths.data / "Summary_MR_SIA.csv",
                          header=1,
                          usecols=["Country","Start date","End date",
                                    "Target population","Reached population"],
                          )
    sia_cal["time"] = sia_cal["Start date"].str.replace("-", "/15/20").map(lambda s: pd.to_datetime(s, errors="coerce"))    
    # sia_cal["time"] = pd.to_datetime(sia_cal["Start date"]\
                        # .str.replace("-","/15/20"),
                    # errors="coerce")
    #sia_cal["time"] = sia_cal["time"].fillna(
    #    pd.to_datetime(sia_cal["Start date"].str.replace("-","/15/20"),
    #        errors="coerce")
    #    )
    sia_cal["doses"] = pd.to_numeric(sia_cal["Reached population"]\
                                    .str.replace(" ",""))
    sia_cal["doses"] = sia_cal["doses"].fillna(
        pd.to_numeric(sia_cal["Target population"]\
                                    .str.replace(" ","")))

    ## Fit the seasonality model
    logt = LogNormal(data)
    i_to_c = {i:c for i,c in enumerate(logt.ln_cr.columns)}
    
    ## Compute high-season months
    outbreak_months = get_outbreak_months(logt.p_low,
                        key_conv=i_to_c)[countries_list[0]]
    print(outbreak_months)

    ## Compute demographic pressure parameters via
    ## endemic stability
    stability_extent = 5 #int((len(data)/12 + 1))
    A0 = np.tril(np.ones((12,12)),k=-1) 
    gf = np.exp(A0 @ (2.*logt.mu_hat[:,0] + logt.sigs[:,0]**2))
    #gf = np.cumprod(np.hstack(stability_extent*[logt.reff[:,0]**2]))
    A1 = np.tril(np.ones((len(gf),len(gf))),k=0)
    A1[:,0] = 0 
    X = np.array([np.ones((len(gf),)),
                  np.arange(len(gf))]).T
    alphas = np.linalg.inv(X.T @ X) @ X.T @ A1 @ gf

    ## Use vaccines and cases to compute sink terms
    cases = data[["time","Cases"]]\
            .copy()\
            .sort_values("time")\
            .set_index("time")["Cases"]
    hist = cases.groupby(lambda t: t.month).sum()
    hist = hist/(hist.sum())
    cases = cases.loc[start_date:]

    ## And vax
    cal = sia_cal.loc[sia_cal["Country"] == countries_list[0]]
    print(cal)
    vax = cal[["time","doses"]]\
            .dropna()\
            .sort_values("time")\
            .set_index("time")["doses"]
    vax = pd.DataFrame(np.diag(vax.values),
                    index=vax.index,columns=vax.index)
    vax = vax.resample("MS").sum()
    vax.index = vax.index + pd.to_timedelta(14,unit="d")
    vax = vax.reindex(cases.index)\
            .fillna(0).sum(axis=1)/1.e6
    num_sias = len(vax.loc[vax != 0])

    ## Set up reconstruction regression
    X = np.array([np.ones((len(cases),)),
                  np.arange(len(cases))]).T
    A1 = np.tril(np.ones((len(cases),len(cases))),k=0)
    A1[:,0] = 0 
    demo_presure = (X @ alphas)[:-1]
    cum_cases = A1[:-1,:-1] @ cases.values[1:]
    cum_vax = A1[:-1,:-1] @ vax.values[:-1]
    if num_sias != 0:
        Xr = np.hstack([cum_cases[:,None],cum_vax[:,None]])
        dXr = np.hstack([cases.values[1:,None],vax.values[:-1,None]])
    else:
        Xr = cum_cases[:,None]
        dXr = cases.values[1:,None]

    ## Solve the problem
    thetaL = np.linalg.inv(Xr.T @ Xr)
    thetas = thetaL @ Xr.T @ demo_presure
    print(thetas)
    theta_cov = (np.sum((demo_presure - (Xr @ thetas))**2)\
            /(len(demo_presure)))*thetaL

    ## Compute the fluctuations and the derivative
    Zt = demo_presure - (Xr @ thetas)
    dZt = alphas[1] - (dXr @ thetas)
    Ztcov = Xr @ theta_cov @ Xr.T
    Zterr = np.sqrt(np.diag(Ztcov))

    ## Get the seasonality model residuals and use them
    ## to rescale the susceptibility estimate
    residual = (logt.ln_cr - logt.X @ logt.mu_hat)\
                .loc[cases.index[:-1]].values.reshape(-1)
    resp = np.exp(residual)-1.
    print(resp.mean())
    print(Zt.mean())
    ls = np.sum(Zt*(np.exp(residual)-1.))/np.sum(Zt**2)
    fig, axes = plt.subplots(figsize=(12,6)) 
    axes.plot(np.exp(residual)-1.)
    axes.plot(ls*Zt)
    fig.tight_layout()
    print(ls)
    Zt *= ls #I0_over_Sbar
    Zterr *= ls #I0_over_Sbar 
    Zt = pd.Series(Zt,index=cases.index[:-1])
    Zterr = pd.Series(Zterr,index=cases.index[:-1])

    ## Extrapolate
    ex_time = pd.date_range(start=cases.index[-1],
                            freq="SMS",periods=2*extrap_months)[::2]
    Zt_extrap = pd.Series(Zt.values[-1]+np.arange(len(ex_time))*alphas[1]*ls,
                          index=ex_time).iloc[1:]
    Zterr_extrap = 0*Zt_extrap + Zterr.values[-1]

    ## Get the high seasons
    high_seasons = pd.Series(Zt_extrap.index.month.isin(outbreak_months),
                      index=Zt_extrap.index).astype(int)
    
    ## Plot the results
    fig = plt.figure(figsize=(15,5))
    axes = fig.add_subplot(1,8,(4,8))
    s_ax = fig.add_subplot(1,8,(1,3))
    axes.plot(Zt,color="k",lw=6,label="Reconstruction",zorder=5)
    axes.plot(Zt-2.*Zterr,color="k",ls="dashed",lw=1,zorder=4)
    axes.plot(Zt+2.*Zterr,color="k",ls="dashed",lw=1,zorder=4)

    ## Plot the forecast
    axes.plot(Zt_extrap,color="C3",lw=3,zorder=4,alpha=0.2)
    axes.plot(Zt_extrap-2.*Zterr_extrap,color="C3",ls="dashed",lw=1,alpha=0.2)
    axes.plot(Zt_extrap+2.*Zterr_extrap,color="C3",ls="dashed",lw=1,alpha=0.2)    
    Zt_extrap.loc[high_seasons != 1] = np.nan
    axes.plot(Zt_extrap,color="C3",lw=6,label="Forecast",zorder=5)
    axes.plot(Zt_extrap-2.*Zterr_extrap,color="C3",ls="dashed",lw=1,zorder=4)
    axes.plot(Zt_extrap+2.*Zterr_extrap,color="C3",ls="dashed",lw=1,zorder=4)
    
    ## Make negative space for timeline stuff
    ylim = axes.get_ylim()
    axes.set_ylim((1.2*ylim[0],ylim[1]))
    ylim = axes.get_ylim()
    axes.axhline(0,color="grey",ls=":",alpha=0.9,lw=3)

    ## Add cases for context    
    axes.fill_between(cases.index,ylim[0],
                      (0.33*(ylim[1]-ylim[0])*(cases.values/cases.max())+ylim[0]),
                      facecolor="grey",edgecolor="None",alpha=0.5,zorder=1)

    ## Add the high seasons
    #axes.fill_between(high_seasons.index,ylim[0],
    #                  (ylim[1]-ylim[0])*high_seasons + ylim[0],
    #                  facecolor="C3",edgecolor="None",alpha=0.5)

    ## Add the campaigns
    sias = vax.loc[vax != 0].index
    for d in sias:
        axes.axvline(d,ymin=0,ymax=0.125,lw=3,color="xkcd:saffron")
    axes.plot([],lw=3,color="xkcd:saffron",label="Campaign",zorder=3)

    # set monthly locator
    axes.xaxis.set_minor_locator(mdates.YearLocator())
    # set formatter
    # axes.xaxis.set_major_formatter(mdates.DateFormatter('%y'))

    ## Tighten
    axes.set_xlabel("Year")
    axes.set_ylim(ylim)
    axes.set_ylabel("Relative susceptibility")
    axes.legend(loc=2,
                #ncol=3,
                frameon=False,
                fancybox=False,
                #bbox_to_anchor=(0.5,-0.1),
                fontsize=18,
                )

    ## Plot the seasonality profile
    ## Create the trig interpolant design matrix
    tk = np.arange(1.,len(logt.mu_hat)+1.)
    t = np.linspace(1.,len(logt.mu_hat)+1.,395)
    dt = (t[:,None]-tk[None,:])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        X_int = np.sin(np.pi*dt)/(12*np.tan(np.pi*dt/12.))
        X_int = np.nan_to_num(X_int,nan=1.)
    mu_hat = X_int @ logt.mu_hat[:,0]
    cov = X_int @ logt.covs[0] @ X_int.T
    sigs = np.sqrt(np.diag(cov)) 
    reff = np.exp(mu_hat + (sigs**2) / 2.0)
    reff_low = np.exp(mu_hat + np.sqrt(2) * sigs * erfinv(2 * 0.1 - 1.0))
    reff_high = np.exp(mu_hat + np.sqrt(2) * sigs * erfinv(2 * 0.9 - 1.0))
    p_low = 0.5 * (1 + erf((-mu_hat) / (sigs * np.sqrt(2))))
    s_ax.fill_between(
        t,
        reff_low,
        reff_high,
        facecolor="k",
        edgecolor="None",
        alpha=0.7,
        zorder=10,
    )
    s_ax.plot(t, reff, color="k", lw=2, alpha=0.9, zorder=20)
    
    ## For a better zoom
    ylim = s_ax.get_ylim()

    ## Plot the data
    hist = pd.concat([hist,
                      pd.Series([hist.loc[1]],index=[13])],
                      axis=0)
    s_ax.bar(hist.index,3*hist.values+ylim[0],
                width=1./5.,
                facecolor="grey",edgecolor="None",
                alpha=0.6,zorder=0.5)
    for m, p in enumerate(logt.p_low[:, 0]):
        if p >= 0.8:  ## Low season threshold
            color = c8c["green"]
        elif (1 - p) >= 0.8:  ## High season threshold
            color = c8c["red"]
        else:  ## indeterminate
            color = c8c["yellow"]
        s_ax.plot(
            [m + 1.]+ ((m+1) == 1)*[m+13.],
            [logt.reff[m, 0]]+((m+1) == 1)*[logt.reff[m, 0]],
            marker="o",
            markersize=13,
            ls="None",
            color=color,
            zorder=40,
        )
    s_ax.axhline(1, color="k", lw=2, zorder=0, ls=":")
    s_ax.set_ylim((ylim[0],1.1*ylim[1]))
    s_ax.set_xticks(np.arange(0, 14, 2))
    s_ax.set_xlabel("Month")
    s_ax.set_xticks(np.arange(1,14,2))
    s_ax.set_xticklabels([i%12 for i in np.arange(1,14,2)])
    s_ax.set_ylabel(r"$R_{eff}(t)$")
    s_ax.text(
        0.05,
        0.9,
        countries_list[0].title(),
        fontsize=26,
        color="k",
        transform=s_ax.transAxes,
    )
    fig.tight_layout()

    ## Add the legend
    #fig.subplots_adjust(bottom=0.15)
    fig.savefig(paths.figures / "fig_3.png")
    plt.show()    