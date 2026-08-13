import csv
import os
import random
from dataclasses import dataclass
from enum import Enum
from functools import partial
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .decoders import PackedXChannelImageDecoder
from .extended import ExtendedVisionDataset

@dataclass
class _AICSMetadata:
    struct : Any

STRUCT = [
    'Lysosome', 'Mitochondria', 'Microtubules', 'Actin filaments',
    'Adherens junctions', 'Centrin-2', 'Gap junctions', 'Desmosomes',
    'Nucleolus (Dense Fibrillar Component)', 'Nuclear envelope',
    'Endoplasmic reticulum', 'Golgi', 'Tight junctions', 'Plasma membrane',
    'Matrix adhesions', 'Peroxisomes', 'Endosomes', 'Actomyosin bundles',
    'Nucleolus (Granular Component)', 'Control - DNA', 'Control - Memb', 'Control - Blank',
    'Control - Noise', 'Control - Random'
    ]