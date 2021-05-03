import pandas as pd
import datetime

df = pd.DataFrame(columns=["date", "pr_factor_to_previous"])


# Running window of twenty weeks
today = datetime.datetime.today()
if today.weekday() == 6:
    data_end = today
else:
    data_end = today - datetime.timedelta(days=today.weekday() + 1)
data_begin = data_end - datetime.timedelta(weeks=12)

for i, day in enumerate(
    pd.date_range(start=data_begin, end=data_end + datetime.timedelta(weeks=10))
):

    if day.weekday() == 6:
        if day.date() == datetime.date(2021, 5, 2):
            df = df.append(
                {"date": day, "pr_factor_to_previous": 0.93}, ignore_index=True
            )
        elif day.date() == datetime.date(2021, 5, 9):
            df = df.append(
                {"date": day, "pr_factor_to_previous": 0.93}, ignore_index=True
            )
        elif day.date() == datetime.date(2021, 5, 16):
            df = df.append(
                {"date": day, "pr_factor_to_previous": 0.93}, ignore_index=True
            )
        else:
            df = df.append({"date": day, "pr_factor_to_previous": 1}, ignore_index=True)


countries = {
    "Austria": {"population": 8933346, "iso2": "AT"},
    "Belgium": {"population": 11396775, "iso2": "BE"},
    "Bulgaria": {"population": 6911167, "iso2": "BG"},
    "Croatia": {"population": 4087234, "iso2": "HR"},
    "Cyprus": {"population": 1213585, "iso2": "CY"},
    "Czechia": {"population": 10723088, "iso2": "CZ"},
    "Denmark": {"population": 5806675, "iso2": "DK"},
    "Estonia": {"population": 1327169, "iso2": "EE"},
    "Finland": {"population": 5546833, "iso2": "FI"},
    "France": {"population": 65376055, "iso2": "FR"},
    "Germany": {"population": 83974064, "iso2": "DE"},
    "Greece": {"population": 10386806, "iso2": "GR"},
    "Hungary": {"population": 9642902, "iso2": "HU"},
    "Iceland": {"population": 342812, "iso2": "IS"},
    "Ireland": {"population": 4976791, "iso2": "IE"},
    "Italy": {"population": 60398612, "iso2": "IT"},
    "Latvia": {"population": 1871316, "iso2": "LV"},
    "Liechtenstein": {"population": 38206, "iso2": "LI"},
    "Lithuania": {"population": 2695154, "iso2": "LT"},
    "Luxembourg": {"population": 633162, "iso2": "LU"},
    "Malta": {"population": 442378, "iso2": "MT"},
    "Netherlands": {"population": 17161789, "iso2": "NL"},
    "Norway": {"population": 5451264, "iso2": "NO"},
    "Poland": {"population": 37817143, "iso2": "PL"},
    "Portugal": {"population": 10175557, "iso2": "PT"},
    "Romania": {"population": 19146259, "iso2": "RO"},
    "Slovakia": {"population": 5461521, "iso2": "SK"},
    "Slovenia": {"population": 2079141, "iso2": "SI"},
    "Spain": {"population": 46767645, "iso2": "ES"},
    "Sweden": {"population": 10143897, "iso2": "SE"},
    "Switzerland": {"population": 8699459, "iso2": "CH"},
    "United Kingdom": {"population": 68138862, "iso2": "GB"},
}

for c in countries:
    df.to_csv(
        f"./data_changepoints/{countries[c]['iso2']}.csv",
        date_format="%Y-%m-%d",
        index=False,
    )
