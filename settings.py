from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import os
from pathlib import Path

BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))

STRATEGIES = {
    "semi-sync": "semi-synchronous",
    "sync": "synchronous",
    "async": "asynchronous"
}

DEFAULT_OUTPUT_DIR = Path('result')
if not DEFAULT_OUTPUT_DIR.is_absolute():
    DEFAULT_OUTPUT_DIR = BASE_DIR.joinpath(DEFAULT_OUTPUT_DIR)
DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CONFIG = Path("configuration/service_1.yml")

if not CONFIG.is_absolute():
    CONFIG = BASE_DIR.joinpath(CONFIG)
