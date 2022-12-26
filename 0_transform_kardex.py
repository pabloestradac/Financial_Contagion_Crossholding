# -*- coding: utf-8 -*-
"""
Created on Fri Apr  6 11:39:00 2018

@author: pablo
"""
# Introduction

"""
Using transform_kardex.py we read the file kardex_accionistas.dta that was received by SUPERCIAS.
The output is the file accionistas.csv that contains firms' equity for each shareholder.
"""

NAME = '0_transform_kardex' ## Name of the notebook goes here (without the file extension!)
PROJECT = '2018_Financial_Contagion_Crossholding'
PYTHON_VERSION = '3.6'


# Preamble


## Imports
"""
All the Python imports go here.
"""
import os, re, math, time
import pandas as pd
import numpy as np

## Set working directory
"""
The code below will traverse the path upwards until it finds the root folder of the project.
"""
workdir = re.sub("(?<={})[\w\W]*".format(PROJECT), "", os.getcwd())
os.chdir(workdir)

## Set  up pipeline folder if missing
"""
The code below will automatically create a pipeline folder for this code file if it does not exist.
"""
if os.path.exists(os.path.join('empirical', '2_pipeline')):
    pipeline = os.path.join('empirical', '2_pipeline', NAME)
else:
    pipeline = os.path.join('2_pipeline', NAME)

if not os.path.exists(pipeline):
    os.makedirs(pipeline)
    for folder in ['out', 'store', 'tmp']:
        os.makedirs(os.path.join(pipeline, folder))


# ---------
# Main code
# ---------

# Defining Variables
year = 2014

# Reading Kardex Data
df_kardex = pd.read_stata(os.path.join('empirical', '0_data', 'external', 'kardex_accionistas.dta'))
df_kardex = df_kardex.astype({'CAM_NOMBRE': 'category', 'KAR_TTRANSACCION': 'category', 'DESCR_TTRANSACCION': 'category',
                              'KAR_TINVERSION': 'category', 'TIPO_INVERSION': 'category'})

# Function to Transform Kardex Data
def shareholderDF(data, year):
    """
    Genera un archivo con los accionistas de cada empresa y la cantidad de capital invertido
    Recibe un dataframe y año
    """
    lista = list(data["EXPEDIENTE"].unique()) # list of unique firms
    data = data.loc[data["FECHA_MOVIMINETO"] <= str(year+1)] # select years below the value
    data = data[['EXPEDIENTE', 'PER_CEDULA', 'KAR_VTOTAL']]
    big_data = pd.DataFrame(columns=['PER_CEDULA', 'KAR_VTOTAL'])
    n = len(lista)
    i = 0

    for empresa in lista:
        i += 1
        example = data[data['EXPEDIENTE']==empresa][['PER_CEDULA', 'KAR_VTOTAL']]
        example = example.groupby('PER_CEDULA').sum()
        example = example.reset_index()
        example = example[example['KAR_VTOTAL']>=0.005]
        example['EXPEDIENTE'] = empresa
        big_data = pd.concat([big_data, example], ignore_index=True)
        print(n - i)

    return big_data

# Generating Shareholder Data
df_shareholder = shareholderDF(df_kardex, year)

# Writing Shareholder Data
df_shareholder.to_csv(os.path.join(pipeline, 'out', 'accionistas{}.csv'.format(year)), index=False)






# ----------
# Leftovers
# ----------
"""
Here you leave any code snippets or temporary code that you don't need but don't want to delete just yet
"""
