# Place all your constants here
import os
import pathlib

import psutil

# ---------------- PATH CONSTANTS ----------------
SRC_PATH = pathlib.Path(__file__).parent
PROJECT_PATH = SRC_PATH.parent
CONFIG_PATH = PROJECT_PATH / "configs"
DATA_PATH = pathlib.Path(os.getenv("DATA_PATH", PROJECT_PATH / "data"))
WEIGHTS_PATH = pathlib.Path(os.getenv("WEIGHTS_PATH", PROJECT_PATH / "genie/weights"))

HYDRA_VERSION_BASE = "1.2"

# --------------- PROJECT CONSTANTS ----------------
CORE_COUNT = psutil.cpu_count(logical=False)
