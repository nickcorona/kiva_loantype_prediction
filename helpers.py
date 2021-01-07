import numpy as np
import pandas as pd


def loguniform(low=0, high=1, size=None, base=10):
    """Returns a number or a set of numbers from a log uniform distribution"""
    return np.power(base, np.random.uniform(low, high, size))


def encode_dates(df, column):
    df.copy()
    df[column + "_year"] = df[column].apply(lambda x: x.year)
    df[column + "_month"] = df[column].apply(lambda x: x.month)
    df[column + "_day"] = df[column].apply(lambda x: x.day)

    df[column + "_hour"] = df[column].apply(lambda x: x.hour)
    df[column + "_minute"] = df[column].apply(lambda x: x.minute)
    df[column + "_second"] = df[column].apply(lambda x: x.second)
    df = df.drop(column, axis=1)
    return df