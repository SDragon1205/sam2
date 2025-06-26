from .misc import *
from .loss import *
from .wandb_callback import *
from .v2v_training_image_preprocessor import *
from .dataset_misc import *
from .eval_function import *
from .heatmap_SAVPE import *
from ultralytics.utils import (YAML, 
                               IterableSimpleNamespace)

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
DEFAULT_CFG_PATH = ROOT / "cfg/default.yaml"
DEFAULT_CFG_DICT = YAML.load(DEFAULT_CFG_PATH)
for k, v in DEFAULT_CFG_DICT.items():
    if isinstance(v, str) and v.lower() == "none":
        DEFAULT_CFG_DICT[k] = None
DEFAULT_CFG_KEYS = DEFAULT_CFG_DICT.keys()
DEFAULT_CFG = IterableSimpleNamespace(**DEFAULT_CFG_DICT)