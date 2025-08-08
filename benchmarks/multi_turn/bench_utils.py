# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import logging
from enum import Enum


class Color(str, Enum):
    RED = "\033[91m"
    GREEN = "\033[92m"
    BLUE = "\033[94m"
    PURPLE = "\033[95m"
    CYAN = "\033[96m"
    YELLOW = "\033[93m"
    RESET = "\033[0m"


TEXT_SEPARATOR = "-" * 100

# Configure the logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] - %(message)s",
    datefmt="%d-%m-%Y %H:%M:%S",
)
logger = logging.getLogger(__name__)
