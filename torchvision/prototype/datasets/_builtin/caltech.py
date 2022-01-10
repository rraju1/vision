import pathlib
import re
from typing import Any, Dict, List, Tuple, BinaryIO

import numpy as np
from torchdata.datapipes.iter import (
    IterDataPipe,
    Mapper,
    Filter,
    IterKeyZipper,
)
from torchvision.prototype.datasets.utils import (
    Dataset,
    DatasetConfig,
    DatasetInfo,
    HttpResource,
    OnlineResource,
    RawImage,
)
from torchvision.prototype.datasets.utils._internal import INFINITE_BUFFER_SIZE, read_mat, hint_sharding, hint_shuffling
from torchvision.prototype.features import Label, BoundingBox, Feature


class Caltech101(Dataset):
    def _make_info(self) -> DatasetInfo:
        return DatasetInfo(
            "caltech101",
            dependencies=("scipy",),
            homepage="http://www.vision.caltech.edu/Image_Datasets/Caltech101",
        )

    def resources(self, config: DatasetConfig) -> List[OnlineResource]:
        images = HttpResource(
            "http://www.vision.caltech.edu/Image_Datasets/Caltech101/101_ObjectCategories.tar.gz",
            sha256="af6ece2f339791ca20f855943d8b55dd60892c0a25105fcd631ee3d6430f9926",
            decompress=True,
        )
        anns = HttpResource(
            "http://www.vision.caltech.edu/Image_Datasets/Caltech101/Annotations.tar",
            sha256="1717f4e10aa837b05956e3f4c94456527b143eec0d95e935028b30aff40663d8",
        )
        return [images, anns]

    _IMAGES_NAME_PATTERN = re.compile(r"image_(?P<id>\d+)[.]jpg")
    _ANNS_NAME_PATTERN = re.compile(r"annotation_(?P<id>\d+)[.]mat")
    _ANNS_CATEGORY_MAP = {
        "Faces_2": "Faces",
        "Faces_3": "Faces_easy",
        "Motorbikes_16": "Motorbikes",
        "Airplanes_Side_2": "airplanes",
    }

    def _is_not_background_image(self, data: Tuple[str, Any]) -> bool:
        path = pathlib.Path(data[0])
        return path.parent.name != "BACKGROUND_Google"

    def _is_ann(self, data: Tuple[str, Any]) -> bool:
        path = pathlib.Path(data[0])
        return bool(self._ANNS_NAME_PATTERN.match(path.name))

    def _images_key_fn(self, data: Tuple[str, Any]) -> Tuple[str, str]:
        path = pathlib.Path(data[0])

        category = path.parent.name
        id = self._IMAGES_NAME_PATTERN.match(path.name).group("id")  # type: ignore[union-attr]

        return category, id

    def _anns_key_fn(self, data: Tuple[str, Any]) -> Tuple[str, str]:
        path = pathlib.Path(data[0])

        category = path.parent.name
        if category in self._ANNS_CATEGORY_MAP:
            category = self._ANNS_CATEGORY_MAP[category]

        id = self._ANNS_NAME_PATTERN.match(path.name).group("id")  # type: ignore[union-attr]

        return category, id

    def _decode_ann(self, data: BinaryIO, image_size: Tuple[int, int]) -> Dict[str, Any]:
        ann = read_mat(data)
        return dict(
            bounding_box=BoundingBox(
                ann["box_coord"].astype(np.int64).squeeze()[[2, 0, 3, 1]], format="xyxy", image_size=image_size
            ),
            contour=Feature(ann["obj_contour"].T),
        )

    def _prepare_sample(
        self, data: Tuple[Tuple[str, str], Tuple[Tuple[str, BinaryIO], Tuple[str, BinaryIO]]]
    ) -> Dict[str, Any]:
        key, (image_data, ann_data) = data
        category, _ = key
        image_path, image_buffer = image_data
        ann_path, ann_buffer = ann_data

        image = RawImage.fromfile(image_buffer)

        return dict(
            self._decode_ann(ann_buffer, image_size=image.probe_image_size()),
            label=Label(self.info.categories.index(category), category=category),
            image_path=image_path,
            image=image,
            ann_path=ann_path,
        )

    def _make_datapipe(
        self,
        resource_dps: List[IterDataPipe],
        *,
        config: DatasetConfig,
    ) -> IterDataPipe[Dict[str, Any]]:
        images_dp, anns_dp = resource_dps

        images_dp = Filter(images_dp, self._is_not_background_image)
        images_dp = hint_sharding(images_dp)
        images_dp = hint_shuffling(images_dp)

        anns_dp = Filter(anns_dp, self._is_ann)

        dp = IterKeyZipper(
            images_dp,
            anns_dp,
            key_fn=self._images_key_fn,
            ref_key_fn=self._anns_key_fn,
            buffer_size=INFINITE_BUFFER_SIZE,
            keep_key=True,
        )
        return Mapper(dp, self._prepare_sample)

    def _generate_categories(self, root: pathlib.Path) -> List[str]:
        resources = self.resources(self.default_config)

        dp = resources[0].load(root)
        dp = Filter(dp, self._is_not_background_image)

        return sorted({pathlib.Path(path).parent.name for path, _ in dp})


class Caltech256(Dataset):
    def _make_info(self) -> DatasetInfo:
        return DatasetInfo(
            "caltech256",
            homepage="http://www.vision.caltech.edu/Image_Datasets/Caltech256",
        )

    def resources(self, config: DatasetConfig) -> List[OnlineResource]:
        return [
            HttpResource(
                "http://www.vision.caltech.edu/Image_Datasets/Caltech256/256_ObjectCategories.tar",
                sha256="08ff01b03c65566014ae88eb0490dbe4419fc7ac4de726ee1163e39fd809543e",
            )
        ]

    def _is_not_rogue_file(self, data: Tuple[str, Any]) -> bool:
        path = pathlib.Path(data[0])
        return path.name != "RENAME2"

    def _prepare_sample(self, data: Tuple[str, BinaryIO]) -> Dict[str, Any]:
        path, buffer = data

        dir_name = pathlib.Path(path).parent.name
        label_str, category = dir_name.split(".")
        return dict(
            path=path,
            image=RawImage.fromfile(buffer),
            label=Label(int(label_str), category=category),
        )

    def _make_datapipe(
        self,
        resource_dps: List[IterDataPipe],
        *,
        config: DatasetConfig,
    ) -> IterDataPipe[Dict[str, Any]]:
        dp = resource_dps[0]
        dp = Filter(dp, self._is_not_rogue_file)
        dp = hint_sharding(dp)
        dp = hint_shuffling(dp)
        return Mapper(dp, self._prepare_sample)

    def _generate_categories(self, root: pathlib.Path) -> List[str]:
        resources = self.resources(self.default_config)

        dp = resources[0].load(root)
        dir_names = {pathlib.Path(path).parent.name for path, _ in dp}

        return [name.split(".")[1] for name in sorted(dir_names)]
