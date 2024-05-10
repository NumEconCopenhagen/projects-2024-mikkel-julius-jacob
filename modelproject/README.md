# Model analysis project

Our project is titled **Relationship and Causality between interest rates and inflation in the US** and is about how interest rates and inflation affect each other and about the causality in the relationship. 

The **results** of the project can be seen from running [modelproject.ipynb](modelproject.ipynb).

**Dependencies:** Apart from a standard Anaconda Python 3 installation, the project requires the following packages, which can imported as:

import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.api import VAR
from scipy.stats import pearsonr
from statsmodels.tsa.statespace.varmax import VARMAX
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from statsmodels.stats.diagnostic import acorr_ljungbox, het_breuschpagan
from statsmodels.stats.stattools import jarque_bera
from statsmodels.stats.sandwich_covariance import cov_hac_simple
from sklearn.utils import resample
from sklearn.linear_model import LinearRegression