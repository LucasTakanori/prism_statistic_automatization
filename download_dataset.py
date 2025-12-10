import os
from pathlib import Path

import synapseclient
import synapseutils

PROJECT_ROOT = Path("/home/lsanc68/ece_bst_link/lsanc68/VOC_ALS")
DATASET_DIR = PROJECT_ROOT / "data"
DATASET_DIR.mkdir(parents=True, exist_ok=True)

auth_token = os.getenv("SYNAPSE_AUTH_TOKEN")
if not auth_token:
    raise RuntimeError("Set SYNAPSE_AUTH_TOKEN environment variable before running.")

syn = synapseclient.Synapse()
syn.login(authToken=auth_token, silent=True)