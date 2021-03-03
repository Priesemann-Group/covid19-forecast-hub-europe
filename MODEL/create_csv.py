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
        df = df.append({"date": day, "pr_factor_to_previous": 1}, ignore_index=True)

df.to_csv("change_points.csv", date_format="%Y-%m-%d", index=False)
