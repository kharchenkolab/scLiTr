from __future__ import annotations

from scanpy import read, AnnData
from pathlib import Path

Zenodo_record = "11401394"

def Weinreb_in_vitro(
    file_path: str | Path = "data/Weinreb_in_vitro.h5ad",
) -> AnnData:
    """
    Dataset from [PMID: 31974159] with in vitro hematopoiesis.

    Parameters
    ----------
    file_path : str | Path, optional
        Path where .h5ad-container will be stored, by default "data/Weinreb_in_vitro.h5ad".

    Returns
    -------
    AnnData
        Annotated data matrix with the dataset.
    """
    url = f"https://zenodo.org/records/{Zenodo_record}/files/Weinreb_in_vitro.h5ad"
    
    adata = read(file_path, backup_url=url, sparse=True, cache=True)
    return adata

def Erickson_murine_development(
    perturbed: bool = False,
    region: str = "trunk",
    subset: str = "all",
    file_path: str | Path | None = None,
) -> AnnData:
    """
    Dataset from [PMID: None] with clonal atlas of murine development.

    Parameters
    ----------
    perturbed : bool, optional
        If the dataset should contain also experiments with mosaic knockouts,
        by default False.
    region : str, optional
        Region of interest, one of `trunk` or `head`, by default "trunk".
    subset : str, optional
        Subset of interest (`all`,  `neurons` or `mesenchyme`), by default "all".
    file_path : str | Path | None, optional
        Path where .h5ad-container will be stored, by default None.

    Returns
    -------
    AnnData
        Annotated data matrix with the dataset
    """
    datasets = {
        False: {
            "trunk": {
                "all": "Erickson_trunk.h5ad",
                "mesenchyme": "Erickson_trunk_mesenchyme.h5ad",
                "neurons": "Erickson_trunk_neurons.h5ad",
                "neural crest": "Erickson_trunk_NC.h5ad",
            },
            "head": {
                "all": "Erickson_head.h5ad",
                "mesenchyme": "Erickson_head_mesenchyme.h5ad",
                "neurons": "Erickson_head_neurons.h5ad",
            },
        },
        True: {
            "trunk": {
                "all": "Erickson_trunk_perturbed.h5ad",
                "mesenchyme": "Erickson_trunk_mesenchyme_perturbed.h5ad",
                "neurons": "Erickson_trunk_neurons_perturbed.h5ad",
                "neural crest": "Erickson_trunk_NC_perturbed.h5ad",
            },
            "head": {
                "all": "Erickson_head_perturbed.h5ad",
                "mesenchyme": "Erickson_head_mesenchyme_perturbed.h5ad",
                "neurons": "Erickson_head_neurons_perturbed.h5ad",
            },
        },
    }
    
    dataset = datasets[perturbed][region][subset]
    
    if file_path is None:
        file_path = f"data/{dataset}"
        
    url = f"https://zenodo.org/records/{Zenodo_record}/files/{dataset}"
    
    adata = read(file_path, backup_url=url, sparse=True, cache=True)
    return adata