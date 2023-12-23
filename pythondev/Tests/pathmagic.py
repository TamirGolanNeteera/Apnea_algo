# Copyright (c) 2020 Neteera Technologies Ltd. - Confidential

"""Path hack to make tests work."""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
