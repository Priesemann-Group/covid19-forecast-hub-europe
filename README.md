# European COVID-19 Forecast Hub [Submission Fork]

This Fork is used to submit weekly updates to the ForecastHub-Eu project. The initial/main repo can be found [here](https://github.com/epiforecasts/covid19-forecast-hub-europe).

## Bayesian inference of SIR-dynamics

- Our model is based on the research article available on __[arXiv](https://arxiv.org/abs/2004.01105)__ and is in press __[at Science](https://science.sciencemag.org/content/early/2020/05/14/science.abb9789)__. In addition we published technical notes, answering some common questions: __[technical notes](technical_notes_dehning_etal_2020.pdf).__

- We published a python package which we use for our forecasting/modeling approach in this repository: __[covid19_inference](https://github.com/Priesemann-Group/covid19_inference)__

- Additional you can find daily updated figures for current numbers in Germany __[here](https://github.com/Priesemann-Group/covid19_inference_forecast)__.


## Model

This model simulates SIR-dynamics with a log-normal convolutions of infections to obtain the delayed reported cases.
Parameters of the model are sampled with Hamiltonian Monte-Carlo using the PyMC3 Python library. We assume that the infection rate can change every week, with a standard deviation that is also an optimized parameter. When new governmental restrictions are enacted or lifted, we include a small prior to the change of the infection rate.

The scripts to run our model can be found in the `MODEL` folder.