""" LogNormal.py
From Thakkar et al. (2024)

Using linear regression to estimate a log-normal seasonality profile from time series 
data, and using that distribution to classify high and low transmission months.
"""

## Standard imports
import numpy as np
import pandas as pd

## For the log normal cdf, quantiles
from scipy.special import erf, erfinv

class LogNormal:
    def __init__(self, tr_data, start_date="2014-01-01"):
        self.alphas = None
        self.Zt = None
        self.Zterr = None
        self.dZt = None
        self.tr_data = tr_data.copy()

        ## Make two data frames by country, one for cases today, one
        ## for datas next month
        Cm = tr_data.pivot(index="time", columns="Country", values="Cases")
        Cm_1 = tr_data.pivot(index="time", columns="Country", values="cases_next_month")
        ln_cr = 0.5 * (np.log(Cm_1 + 1) - np.log(Cm + 1))
        self.ln_cr = ln_cr

        ## Make the linear regression operator
        ## Start with the periodic percision matrix
        ## for the prior distribution.
        D2 = (
            np.diag(12 * [-2])
            + np.diag((12 - 1) * [1], k=1)
            + np.diag((12 - 1) * [1], k=-1)
        )
        D2[0, -1] = 1  ## Periodic BCs
        D2[-1, 0] = 1
        pRW2 = np.dot(D2.T, D2) * (
            (2.**4) / 4.0
        )  ## From the total variation of a sine function
        self.pRW2 = pRW2

        ## Then construction the operator mapping beta's to time
        ## stamps
        self.X = np.vstack((int(len(ln_cr) - 1 / len(pRW2)) + 1) * [np.eye(len(pRW2))])[
            : len(ln_cr)
        ]  ## alignment here comes from the start and end months

        ## Then the linear regression operator is
        LR = np.linalg.inv(self.X.T @ self.X + pRW2)

        ## Compute the LR estimates
        self.mu_hat = LR @ self.X.T @ ln_cr.values

        ## Compute the residuals (this is the student's t result)
        RSS = (ln_cr.values - self.X @ self.mu_hat) ** 2
        prior_hat = np.diag(self.mu_hat.T @ pRW2 @ self.mu_hat)
        var = (RSS.sum(axis=0) + prior_hat) / (len(ln_cr) + len(pRW2) - 3)

        ## From which we can compute standard errors
        self.covs = var[:, None, None] * LR[None, :, :]
        self.sigs = np.sqrt(np.diag(LR)[:, None] * var)

        ## Compute the mean and std errors in the effective R
        self.reff = np.exp(self.mu_hat + (self.sigs**2) / 2.0)
        self.reff_err = np.sqrt((np.exp(self.sigs**2) - 1.0)) * self.reff
        self.reff_low = np.exp(self.mu_hat + np.sqrt(2) * self.sigs * erfinv(2 * 0.1 - 1.0))
        self.reff_high = np.exp(self.mu_hat + np.sqrt(2) * self.sigs * erfinv(2 * 0.9 - 1.0))

        ## Compute the low season probabilities
        p_low = 0.5 * (1 + erf((-self.mu_hat) / (self.sigs * np.sqrt(2))))
        self.p_low = p_low

        ## Set dict mapping index to country name
        self.set_i_to_c()

        ## Compute demographic pressure parameters via endemic stability
        self.set_alphas()

        ## Use vaccines and cases to compute sink terms
        cases = self.tr_data[["time","Cases"]]\
                .copy()\
                .sort_values("time")\
                .set_index("time")["Cases"]
        # trim to start date
        self.cases = cases.loc[start_date:]               

    def run_reconstruction(self, cal):
        """ Run reconstruction of relative susceptiblity """

        vax = cal[["time","doses"]]\
                .dropna()\
                .sort_values("time")\
                .set_index("time")["doses"]
        vax = pd.DataFrame(np.diag(vax.values),
                        index=vax.index,columns=vax.index)
        vax = vax.resample("MS").sum()
        vax.index = vax.index + pd.to_timedelta(14,unit="d")
        vax = vax.reindex(self.cases.index)\
                .fillna(0).sum(axis=1)/1.e6
        num_sias = len(vax.loc[vax != 0])

        ## Set up reconstruction regression
        X = np.array([np.ones((len(self.cases),)),
                    np.arange(len(self.cases))]).T
        A1 = np.tril(np.ones((len(self.cases),len(self.cases))),k=0)
        A1[:,0] = 0 
        demo_presure = (X @ self.alphas)[:-1]
        cum_cases = A1[:-1,:-1] @ self.cases.values[1:]
        cum_vax = A1[:-1,:-1] @ vax.values[:-1]
        if num_sias != 0:
            Xr = np.hstack([cum_cases[:,None],cum_vax[:,None]])
            dXr = np.hstack([self.cases.values[1:,None],vax.values[:-1,None]])
        else:
            Xr = cum_cases[:,None]
            dXr = self.cases.values[1:,None]
                    
        ## Solve the problem
        thetaL = np.linalg.inv(Xr.T @ Xr)
        thetas = thetaL @ Xr.T @ demo_presure
        theta_cov = (np.sum((demo_presure - (Xr @ thetas))**2)\
                /(len(demo_presure)))*thetaL

        ## Compute the fluctuations and the derivative
        self.Zt = demo_presure - (Xr @ thetas)
        self.dZt = self.alphas[1] - (dXr @ thetas)
        Ztcov = Xr @ theta_cov @ Xr.T
        self.Zterr = np.sqrt(np.diag(Ztcov))

        ## Get the seasonality model residuals and use them
        ## to rescale the susceptibility estimate
        residual = (self.ln_cr - self.X @ self.mu_hat).loc[self.cases.index[:-1]].values.reshape(-1)
        # resp = np.exp(residual)-1.
        self.ls = np.sum(self.Zt*(np.exp(residual)-1.))/np.sum(self.Zt**2)

        self.Zt *= self.ls #I0_over_Sbar
        self.Zterr *= self.ls #I0_over_Sbar 
        self.Zt = pd.Series(self.Zt,index=self.cases.index[:-1])
        self.Zterr = pd.Series(self.Zterr,index=self.cases.index[:-1])

    def extrapolate(self, extrap_months=24):
        ex_time = pd.date_range(start=self.cases.index[-1],freq="SMS",periods=2*extrap_months)[::2]
        Zt_extrap = pd.Series(self.Zt.values[-1]+np.arange(len(ex_time))*self.alphas[1]*self.ls,index=ex_time).iloc[1:]
        Zterr_extrap = 0*Zt_extrap + self.Zterr.values[-1]
        return (Zt_extrap, Zterr_extrap)

    def set_i_to_c(self):
        # dict mapping index to country name
        self.i_to_c = {i:c for i,c in enumerate(self.ln_cr.columns)}

    def periodic_pad(self, a, length=13):
        """
        Pads the input array `a` periodically to the specified `length`.

        Parameters:
        a (numpy.ndarray): The input array to be padded.
        length (int): The desired length of the padded array. Defaults to 13.

        Returns:
        numpy.ndarray: The padded array with the specified length.
        """
        repeats = int((length / a.shape[0]) + 1)
        return np.vstack(repeats * [a])[:length, :]

    def set_alphas(self):
        """
        Compute demographic pressure parameters via endemic stability
        """

        A0 = np.tril(np.ones((12,12)),k=-1) 
        gf = np.exp(A0 @ (2.*self.mu_hat[:,0] + self.sigs[:,0]**2))
        A1 = np.tril(np.ones((len(gf),len(gf))),k=0)
        A1[:,0] = 0 
        X = np.array([np.ones((len(gf),)),
                    np.arange(len(gf))]).T
        self.alphas = np.linalg.inv(X.T @ X) @ X.T @ A1 @ gf
