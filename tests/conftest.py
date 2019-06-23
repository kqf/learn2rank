import os
import pytest
from model.data import read_dataset
from model.mc import generate_dataset


@pytest.fixture
def path(filename="data/dataset_v2.csv"):
    if not os.path.exists(filename):
        generate_dataset(filename)
    yield filename


@pytest.fixture
def data(path):
    return read_dataset(path)
