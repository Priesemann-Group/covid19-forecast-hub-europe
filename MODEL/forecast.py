"""
    # Script created with typical workflow on jhu data

    Runtime ~ 4h

"""
import argparse


import logging
import datetime
import pandas as pd
import covid19_inference as cov19
import pymc3 as pm
import numpy as np
import pickle
import matplotlib.pyplot as plt
import scipy
import csv
import os

""" Parser i.e. input parameters for the script
"""
parser = argparse.ArgumentParser(description="Run forecast script")
parser.add_argument(
    "-c",
    "--country",
    type=str,
    help="Country string to run forecasting on.",
    required=True,
)
parser.add_argument(
    "-i", "--iso2", type=str, help="ISO 3166-1 alpha-2 of country", required=True,
)

args = parser.parse_args()

log = logging.getLogger(f"ForecastScript [{args.iso2}]")
log.info(f"Running forecast for countries: {args.country}")


""" # Data retrieval
Download JHU data via our own data retrieval module
"""
jhu = cov19.data_retrieval.JHU()
jhu.download_all_available_data(force_local=True)

# Running window of twenty weeks
today = datetime.datetime.today()
if today.weekday() == 6:
    data_end = today
else:
    data_end = today - datetime.timedelta(days=today.weekday() + 1)
data_begin = data_end - datetime.timedelta(weeks=12)


# Get new cases from dataset filtered by date and country
new_cases_obs = jhu.get_new(
    "confirmed", country=args.country, data_begin=data_begin, data_end=data_end
)
total_cases_obs = jhu.get_total(
    "confirmed", country=args.country, data_begin=data_begin, data_end=data_end
)

""" # Create changepoints
"""
cp_fstring = f"./data_changepoints/{args.iso2}.csv"

if not os.path.isfile(cp_fstring):
    df_change_points = None
else:
    df_change_points = pd.read_csv(f"{args.iso2}.csv")
    df_change_points["date"] = pd.to_datetime(
        df_change_points["date"], format="%Y-%m-%d"
    )
    df_change_points = df_change_points.set_index(df_change_points["date"])

change_points = [
    dict(
        pr_mean_date_transient=data_begin - datetime.timedelta(days=1),
        pr_sigma_date_transient=1.5,
        pr_median_lambda=0.12,
        pr_sigma_lambda=0.5,
        pr_sigma_transient_len=0.5,
    ),
]
for day in pd.date_range(start=data_begin, end=data_end + datetime.timedelta(weeks=4)):
    if day.weekday() == 6:

        # Check if dataframe exists:
        if df_change_points is None:
            factor = 1.0
        else:
            # Prior factor to previous
            if day.date() in [i.date() for i in df_change_points.index]:
                index = [i.date() for i in df_change_points.index].index(day.date())
                factor = df_change_points.iloc[index]["pr_factor_to_previous"]
            else:
                log.info(
                    "Changepoint not found in dict using 1 as pr_factor_to_previous"
                )
                factor = 1

        # Add cp
        change_points.append(
            dict(  # one possible change point every sunday
                pr_mean_date_transient=day,
                pr_sigma_date_transient=1.5,
                pr_sigma_lambda=0.2,  # wiggle compared to previous point
                relative_to_previous=True,
                pr_factor_to_previous=factor,
            )
        )

""" # Create model
"""
# Number of days the simulation starts earlier than the data.
# Should be significantly larger than the expected delay in order to always fit the same number of data points.
diff_data_sim = 16

# Forecasthub wants 4 weeks of predictions, to be save let's do 5 weeks
num_days_forecast = 7 * 5
params_model = dict(
    new_cases_obs=new_cases_obs[:],
    data_begin=data_begin,
    fcast_len=num_days_forecast,
    diff_data_sim=diff_data_sim,
    N_population=83e6,
)

# Median of the prior for the delay in case reporting, we assume 10 days
pr_delay = 10

# Create model compartments
with cov19.model.Cov19Model(**params_model) as this_model:

    # Edit pr_sigma_lambda for each cp
    sigma_lambda = pm.HalfStudentT(name="sigma_lambda_cps", nu=4, sigma=0.5)
    for i, cp in enumerate(change_points[1:]):
        cp["pr_sigma_lambda"] = sigma_lambda

    # Create the an array of the time dependent infection rate lambda
    lambda_t_log = cov19.model.lambda_t_with_sigmoids(
        pr_median_lambda_0=0.4,
        pr_sigma_lambda_0=0.5,
        change_points_list=change_points,  # The change point priors we constructed earlier
        name_lambda_t="lambda_t",  # Name for the variable in the trace (see later)
    )

    # set prior distribution for the recovery rate
    mu = pm.Lognormal(name="mu", mu=np.log(1 / 8), sigma=0.2)

    # This builds a decorrelated prior for I_begin for faster inference.
    # It is not necessary to use it, one can simply remove it and use the default argument
    # for pr_I_begin in cov19.SIR
    prior_I = cov19.model.uncorrelated_prior_I(
        lambda_t_log=lambda_t_log,
        mu=mu,
        pr_median_delay=pr_delay,
        name_I_begin="I_begin",
        name_I_begin_ratio_log="I_begin_ratio_log",
        pr_sigma_I_begin=2,
        n_data_points_used=5,
    )

    # Do we want to use SEIR?
    new_cases = cov19.model.SIR(
        lambda_t_log=lambda_t_log,
        mu=mu,
        name_new_I_t="new_I_t",
        name_I_t="I_t",
        name_I_begin="I_begin",
        pr_I_begin=prior_I,
    )

    # Delay the cases by a lognormal reporting delay
    new_cases = cov19.model.delay_cases(
        cases=new_cases,
        name_cases="delayed_cases",
        name_delay="delay",
        name_width="delay-width",
        pr_mean_of_median=pr_delay,
        pr_sigma_of_median=0.2,
        pr_median_of_width=0.3,
    )

    # Modulate the inferred cases by a abs(sin(x)) function, to account for weekend effects
    # Also adds the "new_cases" variable to the trace that has all model features.
    new_cases = cov19.model.week_modulation(
        cases=new_cases,
        name_cases="new_cases",
        name_weekend_factor="weekend_factor",
        name_offset_modulation="offset_modulation",
        week_modulation_type="abs_sine",
        pr_mean_weekend_factor=0.3,
        pr_sigma_weekend_factor=0.5,
        weekend_days=(6, 7),
    )

    # Define the likelihood, uses the new_cases_obs set as model parameter
    cov19.model.student_t_likelihood(new_cases)


""" # MCMC sampling
"""
trace = pm.sample(model=this_model, init="advi", tune=5000, draws=5000)

# Save trace in case there are some problems with post processing
with open(f"./forecast.pickle", "wb") as f:
    pickle.dump((this_model, trace), f)


""" # Data post processing (submission)
We compute the sum of all new cases for the next weeks as defined here:
- https://github.com/epiforecasts/covid19-forecast-hub-europe/wiki/Forecast-format
- Epidemiological Weeks: Each week starts on Sunday and ends on Saturday

Columns in csv
--------------
forecast_date: date 
    Date as YYYY-MM-DD, last day (Monday) of submission window
scenario_id: string, optional
    One of "forecast" or a specified "scenario ID". If this column is not included it will be assumed that its value is "forecast" for all rows
target:  string  
    "# wk ahead inc case" or "# wk ahead inc death" where # is usually between 1 and 4
target_end_date: date
    Date as YYYY-MM-DD, the last day (Saturday) of the target week
location: string
    An ISO-2 country code
type: string
    One of "point" or "quantile"
quantile: numeric
    For quantile forecasts, one of the 23 quantiles in c(0.01, 0.025, seq(0.05, 0.95, by = 0.05), 0.975, 0.99)
value: numeric
    The predicted count, a non-negative number of new cases or deaths in the forecast week

"""
log.info("Starting data post processing for forecasthub submission")
data = pd.DataFrame(
    columns=[
        "forecast_date",
        "scenario_id",
        "target",
        "target_end_date",
        "location",
        "type",
        "quantile",
        "value",
    ]
)
weeks = [0, 1, 2, 3]
quantiles = [
    0.01,
    0.025,
    0.05,
    0.10,
    0.15,
    0.20,
    0.25,
    0.3,
    0.35,
    0.4,
    0.45,
    0.5,
    0.55,
    0.6,
    0.65,
    0.7,
    0.75,
    0.8,
    0.85,
    0.9,
    0.95,
    0.975,
    0.99,
]

for id_week in weeks:
    log.info(f" Forecast week {id_week+1}:")

    # Get target for data start and end
    idx = (data_end.weekday() + 1) % 7  # MON = 0, SUN = 6 -> SUN = 0 .. SAT = 6
    target_start = data_end - datetime.timedelta(days=idx - 7 * id_week)  # sunday
    target_end = target_start + datetime.timedelta(days=6)  # saturday
    log.info(f'\tstart: {target_start.strftime("%Y-%m-%d")}')
    log.info(f'\tend: {target_end.strftime("%Y-%m-%d")}')

    # Get new cases from model and sum them up
    forecast_new_cases, dates = cov19.plot._get_array_from_trace_via_date(
        model=this_model,
        trace=trace,
        var="new_cases",
        start=target_start,
        end=target_end + datetime.timedelta(days=1),
    )

    week_cases = np.median(forecast_new_cases.sum(axis=-1))
    log.info(f"\tnew cases per week: {week_cases}")
    # Add mean datapoint first
    data = data.append(
        {
            "forecast_date": (data_end + datetime.timedelta(days=1)).strftime(
                "%Y-%m-%d"
            ),
            "scenario_id": "forecast",
            "target": f"{str(id_week+1)} wk ahead inc case",
            "target_end_date": target_end.strftime("%Y-%m-%d"),
            "location": "DE",
            "type": "point",
            "quantile": "NA",
            "value": week_cases,
        },
        ignore_index=True,
    )

    for quantile in quantiles:
        # How to do this with sum ? Is this right?
        quantile_cases = np.quantile(forecast_new_cases.sum(axis=-1), quantile)
        log.info(f"\t{quantile:.2f} quantile: {quantile_cases:.0f}")

        # Add quantiles to data
        data = data.append(
            {
                "forecast_date": (data_end + datetime.timedelta(days=1)).strftime(
                    "%Y-%m-%d"
                ),
                "scenario_id": "forecast",
                "target": f"{str(id_week+1)} wk ahead inc case",
                "target_end_date": target_end.strftime("%Y-%m-%d"),
                "location": args.iso2,
                "type": "quantile",
                "quantile": quantile,
                "value": quantile_cases,
            },
            ignore_index=True,
        )


# Save data
fstring = f'../data-processed/Priesemann-bayes/{(data_end + datetime.timedelta(days=1)).strftime("%Y-%m-%d")}-Priesemann-bayes.csv'


# If file does not exist create header
# It could happen that files gets overwritten with this setup but is very unlikely... Good for now
if not os.path.isfile(fstring):
    with open(fstring, "wb") as file:
        data.to_csv(
            file, header=True, index=False, quoting=csv.QUOTE_ALL,
        )
else:
    # Append to existing file
    with open(fstring, "ab") as file:
        data.to_csv(
            file, mode="a", header=False, index=False, quoting=csv.QUOTE_ALL,
        )

""" # Sanity check plots
"""
save_to = f"./figures/{args.iso2}_"

fig, axes = cov19.plot.timeseries_overview(this_model, trace, offset=total_cases_obs[0])

for ax in axes:
    ax.set_xlim(datetime.datetime.now() - datetime.timedelta(days=4 * 17))
# Set y lim for effective growth rate
axes[0].set_ylim(-0.08, 0.1)
axes[1].set_ylim(0, new_cases_obs.max() + 5000)
# Add vline for today

""" Add text for current reproduction number
"""


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2.0, n - 1)
    return m, m - h, m + h


f_trunc = lambda number, precision: "{{:.{}f}}".format(precision).format(number)
num_rows = len(change_points) + 1 + 1
last = num_rows - 2
current_R = (trace[f"lambda_{last}"] - trace["mu"] + 1) ** 4
from covid19_inference.plot import (
    _get_mpl_text_coordinates,
    _add_mpl_rect_around_text,
)

med, perc1, perc2 = mean_confidence_interval(current_R)
text_md = f"{f_trunc(med,3)}"
text_ci = f"[{f_trunc(perc1,3)}, {f_trunc(perc2,3)}]"
tel_md = axes[0].text(
    0.8,
    0.9,  # let's have a ten percent margin or so
    r"Current $R_{RKI} \simeq " + text_md + r"$",
    fontsize=10,
    transform=axes[0].transAxes,
    verticalalignment="top",
    horizontalalignment="center",
    zorder=100,
)
x_min, x_max, y_min, y_max = _get_mpl_text_coordinates(tel_md, axes[0])
tel_ci = axes[0].text(
    0.8,
    y_min * 0.9,  # let's have a ten percent margin or so
    text_ci,
    fontsize=8,
    transform=axes[0].transAxes,
    verticalalignment="top",
    horizontalalignment="center",
    zorder=101,
)
_add_mpl_rect_around_text(
    [tel_md, tel_ci], axes[0], facecolor="#F0F0F0", alpha=0.5, zorder=99,
)

# --------------------------------------------------------------------------- #
# inset new cases
# --------------------------------------------------------------------------- #
# Add inset for march to juli
"""
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes

axins = axes[1].inset_axes(bounds=[0.1, 0.5, 0.4, 0.4])
for line in axes[1].lines:
    axins.lines.append(line)

ax = axins

new_cases_inset = owd.get_new(
    "confirmed",
    country="Germany",
    data_begin=datetime.datetime(2020, 4, 2),
    data_end=data_end,
)

# model fit
cov19.plot._timeseries(
    x=new_cases_inset.index, y=new_cases_inset, ax=ax, what="model", color="tab:orange",
)
prec = 1.0 / (np.log10(ax.get_ylim()[1]) - 2.5)
if prec < 2.0 and prec >= 0:
    ax.yaxis.set_major_formatter(
        mpl.ticker.FuncFormatter(cov19.plot._format_k(int(prec)))
    )
    ticks = ax.get_xticks()
    ax.set_xticks(ticks=[new_cases_inset.index.min(), new_cases_inset.index.max()])

"""
# save: ts for timeseries
plt.savefig(
    save_to + "ts.png", dpi=300, bbox_inches="tight", pad_inches=0.05,
)
