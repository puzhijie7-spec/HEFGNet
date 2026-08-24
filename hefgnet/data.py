"""Datasets and paired image-mask transforms used by HEFGNet."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode, RandomCrop
from torchvision.transforms import functional as TF


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _resolve_dataset_root(root: str | Path, candidates: tuple[str, ...]) -> Path:
    root = Path(root).expanduser()
    if root.exists() and any((root / marker).exists() for marker in candidates):
        return root
    for name in candidates:
        candidate = root / name
        if candidate.exists():
            return candidate
    names = ", ".join(candidates)
    raise FileNotFoundError(
        f"Could not locate a dataset under {root}. Expected the dataset directory "
        f"itself or one of these subdirectories: {names}."
    )


def _read_ids(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(f"Dataset split file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return [Path(line.strip()).stem for line in handle if line.strip()]


def _load_pair(image_path: Path, mask_path: Path) -> tuple[Image.Image, Image.Image]:
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not mask_path.is_file():
        raise FileNotFoundError(f"Mask not found: {mask_path}")
    return Image.open(image_path).convert("RGB"), Image.open(mask_path)


class SegmentationTransform:
    """Apply the paired augmentations used in the reported experiments."""

    def __init__(
        self,
        image_size: int = 768,
        training: bool = False,
        horizontal_flip_probability: float = 0.6,
        cutmix_probability: float = 0.5,
        cutmix_beta: float = 1.0,
    ) -> None:
        self.image_size = image_size
        self.training = training
        self.horizontal_flip_probability = horizontal_flip_probability
        self.cutmix_probability = cutmix_probability
        self.cutmix_beta = cutmix_beta

    @staticmethod
    def _pad_to_crop(
        image: Image.Image,
        mask: Image.Image,
        crop_size: int,
    ) -> tuple[Image.Image, Image.Image]:
        width, height = image.size
        pad_right = max(crop_size - width, 0)
        pad_bottom = max(crop_size - height, 0)
        if pad_right or pad_bottom:
            padding = (0, 0, pad_right, pad_bottom)
            image = TF.pad(image, padding, fill=0)
            mask = TF.pad(mask, padding, fill=255)
        return image, mask

    def _training_geometry(
        self,
        image: Image.Image,
        mask: Image.Image,
    ) -> tuple[Image.Image, Image.Image]:
        resize_to = random.randint(self.image_size // 2, self.image_size * 2)
        image = TF.resize(
            image,
            resize_to,
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        )
        mask = TF.resize(mask, resize_to, interpolation=InterpolationMode.NEAREST)

        if random.random() < self.horizontal_flip_probability:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        image, mask = self._pad_to_crop(image, mask, self.image_size)
        top, left, height, width = RandomCrop.get_params(
            image, (self.image_size, self.image_size)
        )
        image = TF.crop(image, top, left, height, width)
        mask = TF.crop(mask, top, left, height, width)
        return image, mask

    def _evaluation_geometry(
        self,
        image: Image.Image,
        mask: Image.Image,
    ) -> tuple[Image.Image, Image.Image]:
        output_size = [self.image_size, self.image_size]
        image = TF.resize(
            image,
            output_size,
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        )
        mask = TF.resize(mask, output_size, interpolation=InterpolationMode.NEAREST)
        return image, mask

    @staticmethod
    def _cutmix_box(size: int, ratio: float) -> tuple[int, int, int, int]:
        cut_ratio = np.sqrt(1.0 - ratio)
        cut_width = int(size * cut_ratio)
        cut_height = int(size * cut_ratio)
        center_x = np.random.randint(size)
        center_y = np.random.randint(size)
        x1 = int(np.clip(center_x - cut_width // 2, 0, size))
        y1 = int(np.clip(center_y - cut_height // 2, 0, size))
        x2 = int(np.clip(center_x + cut_width // 2, 0, size))
        y2 = int(np.clip(center_y + cut_height // 2, 0, size))
        return x1, y1, x2, y2

    def _to_tensor(
        self,
        image: Image.Image,
        mask: Image.Image,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        image_tensor = TF.to_tensor(image)
        image_tensor = TF.normalize(image_tensor, IMAGENET_MEAN, IMAGENET_STD)
        mask_array = np.asarray(mask)
        if mask_array.ndim == 3:
            mask_array = mask_array[..., 0]
        mask_tensor = torch.as_tensor(mask_array.copy(), dtype=torch.long)
        return image_tensor, mask_tensor

    def __call__(
        self,
        image: Image.Image,
        mask: Image.Image,
        pair_loader: Callable[[], tuple[Image.Image, Image.Image]] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.training:
            image, mask = self._evaluation_geometry(image, mask)
            return self._to_tensor(image, mask)

        random_state = random.getstate()
        image, mask = self._training_geometry(image, mask)
        if pair_loader is not None and random.random() < self.cutmix_probability:
            second_image, second_mask = pair_loader()
            random.setstate(random_state)
            second_image, second_mask = self._training_geometry(
                second_image, second_mask
            )
            ratio = np.random.beta(self.cutmix_beta, self.cutmix_beta)
            x1, y1, x2, y2 = self._cutmix_box(self.image_size, ratio)
            image_array = np.asarray(image).copy()
            mask_array = np.asarray(mask).copy()
            second_image_array = np.asarray(second_image)
            second_mask_array = np.asarray(second_mask)
            image_array[y1:y2, x1:x2] = second_image_array[y1:y2, x1:x2]
            mask_array[y1:y2, x1:x2] = second_mask_array[y1:y2, x1:x2]
            image = Image.fromarray(image_array)
            mask = Image.fromarray(mask_array)
        return self._to_tensor(image, mask)


class PairedSegmentationDataset(Dataset):
    def __init__(
        self,
        images: list[Path],
        masks: list[Path],
        transform: SegmentationTransform,
    ) -> None:
        if len(images) != len(masks) or not images:
            raise ValueError("The image and mask lists must be non-empty and aligned.")
        self.images = images
        self.masks = masks
        self.transform = transform

    def __len__(self) -> int:
        return len(self.images)

    def _load_index(self, index: int) -> tuple[Image.Image, Image.Image]:
        return _load_pair(self.images[index], self.masks[index])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image, mask = self._load_index(index)

        def load_random_pair() -> tuple[Image.Image, Image.Image]:
            return self._load_index(random.randrange(len(self)))

        pair_loader = load_random_pair if self.transform.training else None
        return self.transform(image, mask, pair_loader)


def build_foodseg103(
    data_root: str | Path,
    split: str,
    transform: SegmentationTransform,
) -> PairedSegmentationDataset:
    root = _resolve_dataset_root(data_root, ("Images", "FoodSeg103"))
    if root.name != "FoodSeg103" and (root / "FoodSeg103").is_dir():
        root = root / "FoodSeg103"
    subset = "train" if split == "train" else "test"
    identifiers = _read_ids(root / "ImageSets" / f"{split}.txt")
    image_dir = root / "Images" / "img_dir" / subset
    mask_dir = root / "Images" / "ann_dir" / subset
    images = [image_dir / f"{identifier}.jpg" for identifier in identifiers]
    masks = [mask_dir / f"{identifier}.png" for identifier in identifiers]
    return PairedSegmentationDataset(images, masks, transform)


def build_uecfoodpixcomplete(
    data_root: str | Path,
    split: str,
    transform: SegmentationTransform,
) -> PairedSegmentationDataset:
    root = _resolve_dataset_root(
        data_root, ("img", "UEC", "UECFoodPixComplete")
    )
    if not (root / "img").is_dir():
        for name in ("UEC", "UECFoodPixComplete"):
            if (root / name / "img").is_dir():
                root = root / name
                break
    split_file = "train9000.txt" if split == "train" else "test1000.txt"
    identifiers = _read_ids(root / split_file)
    image_dir = root / "img" / "all"
    mask_dir = root / "mask"
    images = [image_dir / f"{identifier}.jpg" for identifier in identifiers]
    masks = [mask_dir / f"{identifier}.png" for identifier in identifiers]
    return PairedSegmentationDataset(images, masks, transform)


def build_dataset(
    name: str,
    data_root: str | Path,
    split: str,
    image_size: int = 768,
) -> PairedSegmentationDataset:
    if split not in {"train", "test"}:
        raise ValueError("split must be either 'train' or 'test'.")
    transform = SegmentationTransform(
        image_size=image_size,
        training=split == "train",
    )
    normalized_name = name.lower().replace("-", "").replace("_", "")
    if normalized_name == "foodseg103":
        return build_foodseg103(data_root, split, transform)
    if normalized_name in {"uec", "uecfoodpixcomplete"}:
        return build_uecfoodpixcomplete(data_root, split, transform)
    raise ValueError(f"Unsupported dataset: {name}")
