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

DEFAULT_OUTPUT_DIR = 'result'
