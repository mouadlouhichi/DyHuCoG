import torch, random, numpy as np, logging, yaml, os, pathlib

_LOG_FORMAT = "%(asctime)s | %(levelname)s | %(message)s"
logging.basicConfig(level=logging.INFO, format=_LOG_FORMAT, datefmt="%H:%M:%S")
logger = logging.getLogger("dyhucog")

def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info(f"Seeds set to {seed}")

def load_cfg(cfg_path: str):
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
