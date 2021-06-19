import pandas as pd
from os import path
import numpy as np

def fn_row_norm(an_array):
    """
    Row-normalizing of array
    """
    sum_of_rows = an_array.sum(axis=1)
    return an_array / sum_of_rows[:, np.newaxis]

def get_dataset(dataname):
    """
    Download and return dataset

    :param str dataname: Name of dataset
    :return: pandas.DataFrame containing the data
    """
    datafile = path.join(path.dirname(__file__), f'simulated/{dataname}.csv')
    w_file = path.join(path.dirname(__file__),'simulated/weight_matrix_N025.csv')
    data = pd.read_csv(datafile)
    w_mat = pd.read_csv(w_file, header = None)
    m_W0 = np.array(w_mat)
    m_W = fn_row_norm(m_W0)

    return (data,m_W)


def get_sample_data():
    """
    Function that returns sample data.
    :return typle(pandas.DataFrame,numpy.array): (data,m_W)
    """
    return get_dataset('data_dynamic1_gaussian0_N025_T200')
