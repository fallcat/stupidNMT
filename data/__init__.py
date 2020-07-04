'''
Initialize the data module
'''
from functools import partial

from data.annotated import TextAnnotation
from data.iwslt import IWSLTDataset, IWSLTEnJpDataset
from data.wmt import WMTEnDeDataset, WMTEnFrDataset, WMTEnFrFullDataset, WMTEnRoDataset

DATASETS = {
    _dataset.name(_swap, _annotation): partial(_dataset, swap=_swap, annotation=_annotation)
    for _dataset in (WMTEnDeDataset, WMTEnFrDataset, WMTEnFrFullDataset, IWSLTDataset, WMTEnRoDataset, IWSLTEnJpDataset)
    for _annotation in TextAnnotation
    for _swap in [False, True]
}

