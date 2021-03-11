# ------------------------------------------------------------------------------ #
# @Author:        Sebastian B. Mohr
# @Email:
# @Created:       2021-03-11 15:55:15
# @Last Modified: 2021-03-11 16:02:54
# ------------------------------------------------------------------------------ #
import covid19_inference as cov19

jhu = cov19.data_retrieval.JHU()
jhu.download_all_available_data(force_download=True)
