from os.path import join as pjoin
from dipy.data.fetcher import _make_fetcher, dipy_home

dname = '../data/' # or dipy_home

fetch_sherby = _make_fetcher(
    "fetch_sherby",
    pjoin(dname, 'sherby'),
    'https://dl.dropboxusercontent.com/u/2481924/',
    ['Sherby.zip'],
    ['Sherby.zip'],
    ['2979482087f5e37846e802ea19542d52'],
    doc="Download 2 subjects with DWI and T1 datasets",
    data_size="200MB",
    unzip=True)

fetch_sherby()
