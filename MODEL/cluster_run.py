# ------------------------------------------------------------------------------ #
# @Author:        Sebastian B. Mohr
# @Email:
# @Created:       2021-03-11 14:52:21
# @Last Modified: 2021-03-11 15:25:50
# ------------------------------------------------------------------------------ #
import argparse
import logging
import os
from multiprocessing import Pool

log = logging.getLogger("ClusterRunner")

parser = argparse.ArgumentParser(description="Run forecast script")
parser.add_argument(
    "-i", "--id", type=int, help="ID", required=True,
)

args = parser.parse_args()
log.info(f"ID: {args.id}")


countries = {
    "Austria": {"iso2": "AT"},
    "Belgium": {"iso2": "BE"},
    "Bulgaria": {"iso2": "BY"},
    "Croatia": {"iso2": "HR"},
    "Cyprus": {"iso2": "CY"},
    "Czechia": {"iso2": "CZ"},
    "Denmark": {"iso2": "DK"},
    "Estonia": {"iso2": "EE"},
    "Finland": {"iso2": "FI"},
    "France": {"iso2": "FX"},
    "Germany": {"iso2": "DE"},
    "Greece": {"iso2": "GR"},
    "Hungary": {"iso2": "HU"},
    "Iceland": {"iso2": "IS"},
    "Ireland": {"iso2": "IE"},
    "Italy": {"iso2": "IT"},
    "Latvia": {"iso2": "LV"},
    "Liechtenstein": {"iso2": "LI"},
    "Lithuania": {"iso2": "LT"},
    "Luxembourg": {"iso2": "LU"},
    "Malta": {"iso2": "MT"},
    "Netherlands": {"iso2": "NL"},
    "Norway": {"iso2": "NO"},
    "Poland": {"iso2": "PL"},
    "Portugal": {"iso2": "PT"},
    "Romania": {"iso2": "RO"},
    "Slovakia": {"iso2": "SK"},
    "Slovenia": {"iso2": "SI"},
    "Spain": {"iso2": "ES"},
    "Sweden": {"iso2": "SE"},
    "Switzerland": {"iso2": "CH"},
    "United Kingdom": {"iso2": "GB"},
}

# Split countries into groups of 8
mapping = []
inner = []
count = 0
for i, country in enumerate(countries):

    inner.append(country)

    if count == 8:
        mapping.append(inner)
        inner = []
        count = 0
    count += 1
mapping.append(inner)


def exec(country_string):
    """
    Executes python script
    """
    os.system(
        f'python forecast.py -c {country_string} -i {countries[country_string]["iso2"]}'
    )


with Pool(32 // 4) as p:
    p.map(exec, mapping[args.id])
