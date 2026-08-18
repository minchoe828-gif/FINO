import os
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional, List
import os
import csv

from .decoders import ImageDataDecoder
from .extended import ExtendedVisionDataset

@dataclass
class _AICSMetadata:
    struct : Any

class _Split(Enum):
    TRAIN_SSL = 'train_ssl'

STRUCT = [
    'Lysosome', 'Mitochondria', 'Microtubules', 'Actin filaments',
    'Adherens junctions', 'Centrin-2', 'Gap junctions', 'Desmosomes',
    'Nucleolus (Dense Fibrillar Component)', 'Nuclear envelope',
    'Endoplasmic reticulum', 'Golgi', 'Tight junctions', 'Plasma membrane',
    'Matrix adhesions', 'Peroxisomes', 'Endosomes', 'Actomyosin bundles',
    'Nucleolus (Granular Component)', 'Control - DNA', 'Control - Memb', 'Control - Blank',
    'Control - Noise', 'Control - Random'
]

_struct_to_ids = {struct: idx for idx, struct in enumerate(STRUCT)}

EXTERNAL_CSV_RELPATH = "metadata.csv"

def _load_file_names_and_labels_ssl(root: str):
    #metadata_by_filename: 
    csv_relpath = EXTERNAL_CSV_RELPATH

    image_paths: List[str] = []
    labels: List[int] = []
    struct_ids: List[int] = []

    with open(os.path.join(root, csv_relpath)) as f:
        for row in csv.DictReader(f):
            image_relpath = row.get("save_reg_path", "")
            if image_relpath:
                image_paths.append(image_relpath)
            struct_displayname = row.get("StructureDisplayName", "")
            if struct_displayname:
                struct_ids.append(_struct_to_ids.get(struct_displayname, ""))

    labels = list(range(len(image_paths)))

    return image_paths, labels, struct_ids

class AICSDataset(ExtendedVisionDataset):
    """Allen Institute pipeline_integrated_single_cell dataset 
    
    Args:
        split: TRAIN_SSL
        with_metadata: When True(default), ``__getitem`` returns
            ``(image, (label, _AICSMetadata))``.
        root: Directory containing metadata.csv
    """

    Split = _Split
    Metadata = _AICSMetadata

    def __init__(
            self,
            *,
            split: _Split=_Split.TRAIN_SSL,
            with_metadata: bool=True,
            root: str,
            **kwargs: Any,
    ) -> None:
        super.__init__(
                image_decoder= ImageDataDecoder(),
                root=root,
                **kwargs,
        )

        self._split = _Split(split) if isinstance(split, str) else split
        self._with_metadata = with_metadata
        self._struct_ids: Optional[List[int]] = None

        if self._split == _Split.TRAIN_SSL:
            (
                self._image_relpaths, self._labels,
                self._struct_ids,
            ) = _load_file_names_and_labels_ssl()

        def get_image_relpath(self, index: int) -> str:
            return self._image_relpaths[index]

        def get_image_data(self, index: int) -> bytes:
            with open(os.path.join(root, get_image_relpath[index]), "rb") as f:
                return f.read()

        def get_target(self, index: int) -> Any:
            target = self._labels[index]

            if not with_metadata:
                return target

            metadata = _AICSMetadata(
                struct=self._struct_ids[index]
            )
            return (target, metadata)

        def __len__(self) -> int:
            return len(self._image_relpaths)




