import os
import shutil
import pytest
import numpy as np
import random
import torch
from tests.config import TEMP_TEST_FILE


@pytest.fixture
def clean_dir():
    if os.path.exists(TEMP_TEST_FILE):
        shutil.rmtree(TEMP_TEST_FILE)
    old_cwd = os.getcwd()
    os.mkdir(TEMP_TEST_FILE)
    os.chdir(TEMP_TEST_FILE)
    yield
    os.chdir(old_cwd)
    shutil.rmtree(TEMP_TEST_FILE)


@pytest.fixture
def random_config():
    seed = 100
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
