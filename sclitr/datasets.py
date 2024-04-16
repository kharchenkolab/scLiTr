from __future__ import annotations

from scanpy import read, AnnData
from pathlib import Path

def Weinreb_in_vitro(
    file_path: str | Path = "data/Weinreb_in_vitro.h5ad",
) -> AnnData:
    url = ""
    adata = read(file_path, backup_url=url, sparse=True, cache=True)
    return adata
