# ------------------------------------------------------------------------------ #
# @Author:        Sebastian B. Mohr
# @Email:
# @Created:       2021-03-11 14:52:21
# @Last Modified: 2021-03-18 11:38:58
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
args.id = args.id - 1
log.info(f"ID: {args.id}")


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

# Split countries into groups of 8
num_parallel = 1

mapping = []
inner = []
count = 0
for i, country in enumerate(countries):
    inner.append(country)
    if count == num_parallel:
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
        f'python forecast.py -c {country_string} -i {countries[country_string]["iso2"]} -p {countries[country_string]["population"]}'
    )


with Pool(32 // num_parallel) as p:
    p.map(exec, mapping[args.id])
