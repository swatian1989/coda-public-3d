"""CODA deep learning tissue multi-labelling. Stage 4 of the published protocol.

Implements the Online Methods "Deep learning tissue multi-labelling" section
verbatim:

  - 7 tissue images equally spaced within each sample, manually annotated.
  - Minimum 50 examples of each tissue subtype per image.
  - Stain normalisation of all images in the case against a reference optical
    density, which the paper states is what avoids catastrophic failure of the
    segmentation on unannotated images with different staining.
  - Annotation bounding boxes extracted and saved individually.
  - Training tiles built as 9000 x 9000 x 3 zero-value images, filled by
    randomly overlaying bounding boxes of the LEAST represented class until the
    tile is >65% full and pixels per class are approximately equal.
  - Augmentation: rotation, scaling 0.8-1.2, hue 0.8-1.2 per RGB channel.
  - Each 9000 x 9000 tile cut into 324 tiles of 500 x 500 x 3.
  - 20 large images built, half augmented, giving 6480 training tiles.
    5 more give 1620 validation tiles. 324 testing tiles come from an image
    used for neither training nor validation.
  - ResNet-50 backbone adapted for DeepLab v3+, trained to validation
    patience 5.
  - If >90% per-class precision and recall is not reached, add annotations and
    repeat. That is the acceptance criterion, not a target.
  - Output labelled at 2 um/pixel.

This is the only stage that needs a GPU.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SegmentationConfig:
    """[PAPER] values from the CODA Online Methods."""

    annotation_images_per_sample: int = 7      # [PAPER]
    annotations_per_class: int = 50            # [PAPER] minimum
    big_tile_px: int = 9000                    # [PAPER]
    small_tile_px: int = 500                   # [PAPER] -> 324 per big tile
    fill_fraction: float = 0.65                # [PAPER] >65% full
    n_train_big_tiles: int = 20                # [PAPER] half augmented
    n_val_big_tiles: int = 5                   # [PAPER]
    scale_range: tuple[float, float] = (0.8, 1.2)   # [PAPER]
    hue_range: tuple[float, float] = (0.8, 1.2)     # [PAPER] per RGB channel
    output_mpp: float = 2.0                    # [PAPER]
    validation_patience: int = 5               # [PAPER]
    acceptance_precision: float = 0.90         # [PAPER] per class
    acceptance_recall: float = 0.90            # [PAPER] per class
    backbone: str = "resnet50"                 # [PAPER]
    architecture: str = "deeplabv3plus"        # [PAPER]
    seed: int = 42
    class_names: list[str] = field(default_factory=list)

    @property
    def tiles_per_big(self) -> int:
        return (self.big_tile_px // self.small_tile_px) ** 2   # 324 at 9000/500


def normalise_stain_to_reference(
    rgb: np.ndarray, reference_od: dict[str, np.ndarray],
) -> np.ndarray:
    """Reconstruct an RGB image at a chosen reference optical density. [PAPER]

    The paper is explicit that this step is what prevents catastrophic failure
    of the segmentation on unannotated images whose staining differs from the
    annotated ones. Deconvolve to hematoxylin and eosin concentrations, then
    re-compose using the REFERENCE stain vectors rather than the image's own.
    Concentrations are preserved; only the colour basis changes.
    """
    from .deconv import deconvolve, estimate_stain_vectors

    own = estimate_stain_vectors(rgb)
    conc = deconvolve(rgb, stains=own)

    v = np.stack([reference_od[k] / np.linalg.norm(reference_od[k])
                  for k in ("hematoxylin", "eosin")])
    third = np.cross(v[0], v[1])
    v = np.vstack([v, third / np.linalg.norm(third)])

    h, w = rgb.shape[:2]
    stacked = np.stack([conc["hematoxylin"], conc["eosin"],
                        np.zeros_like(conc["eosin"])], axis=-1)
    od = stacked.reshape(-1, 3) @ v
    out = 255.0 * np.power(10.0, -od)
    return np.clip(out, 0, 255).reshape(h, w, 3).astype(np.uint8)


def build_training_tile(
    annotation_crops: dict[int, list[np.ndarray]],
    cfg: SegmentationConfig,
    augment: bool = False,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Build one 9000 x 9000 training tile by class-balanced overlay. [PAPER]

    The overlay strategy is the part that is easy to get wrong. Boxes of the
    CURRENTLY LEAST REPRESENTED class are placed each iteration, which drives
    pixel counts per class toward equality. Simply placing boxes at random
    reproduces the natural class imbalance of the tissue, and the rare classes
    (islets, nerves, small lesions) then get learned poorly no matter how many
    tiles you build.

    Returns (image, label_mask). Unfilled pixels are label 0 = background.
    """
    rng = rng or np.random.default_rng(cfg.seed)
    n = cfg.big_tile_px
    image = np.zeros((n, n, 3), dtype=np.uint8)
    mask = np.zeros((n, n), dtype=np.int16)

    classes = sorted(annotation_crops)
    pixels = {c: 0 for c in classes}
    filled = 0
    target = cfg.fill_fraction * n * n
    guard = 0

    while filled < target and guard < 200_000:
        guard += 1
        cls = min(classes, key=lambda c: pixels[c])      # least represented
        crops = annotation_crops[cls]
        if not crops:
            classes.remove(cls)
            if not classes:
                break
            continue

        crop = crops[rng.integers(len(crops))]
        if augment:
            crop = _augment(crop, cfg, rng)

        ch, cw = crop.shape[:2]
        if ch >= n or cw >= n:
            continue
        y, x = rng.integers(0, n - ch), rng.integers(0, n - cw)

        region = mask[y:y + ch, x:x + cw]
        free = region == 0
        if free.sum() < 0.5 * free.size:                  # mostly occupied
            continue

        patch = image[y:y + ch, x:x + cw]
        patch[free] = crop[free] if crop.ndim == 3 else crop[free][:, None]
        region[free] = cls
        added = int(free.sum())
        pixels[cls] += added
        filled += added

    frac = filled / (n * n)
    logger.info("tile %.0f%% filled, class pixel spread %.2f (1.0 = perfectly balanced)",
                frac * 100,
                min(pixels.values()) / max(max(pixels.values()), 1))
    if frac < cfg.fill_fraction * 0.9:
        logger.warning("only %.0f%% filled, target was %.0f%%. Too few annotation "
                       "crops, or crops are too small.", frac * 100,
                       cfg.fill_fraction * 100)
    return image, mask


def _augment(crop: np.ndarray, cfg: SegmentationConfig,
             rng: np.random.Generator) -> np.ndarray:
    """Rotation, scaling 0.8-1.2, per-channel hue 0.8-1.2. [PAPER]"""
    from scipy import ndimage

    out = ndimage.rotate(crop, rng.uniform(0, 360), reshape=True, order=1,
                         mode="constant")
    scale = rng.uniform(*cfg.scale_range)
    out = ndimage.zoom(out, (scale, scale, 1) if out.ndim == 3 else scale, order=1)
    if out.ndim == 3:
        for c in range(3):
            out[..., c] = np.clip(out[..., c] * rng.uniform(*cfg.hue_range), 0, 255)
    return out.astype(np.uint8)


def cut_to_small_tiles(
    image: np.ndarray, mask: np.ndarray, cfg: SegmentationConfig,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Cut a 9000 x 9000 tile into 324 tiles of 500 x 500. [PAPER]"""
    s = cfg.small_tile_px
    imgs, masks = [], []
    for y in range(0, image.shape[0] - s + 1, s):
        for x in range(0, image.shape[1] - s + 1, s):
            imgs.append(image[y:y + s, x:x + s])
            masks.append(mask[y:y + s, x:x + s])
    return imgs, masks


def build_dataset(
    annotation_crops: dict[int, list[np.ndarray]], cfg: SegmentationConfig,
) -> dict[str, list]:
    """Full CODA training data generation. [PAPER] 6480 train, 1620 val tiles.

    20 big tiles, half augmented, cut to 500 x 500 gives 20 x 324 = 6480.
    5 more give 1620 validation tiles.
    """
    rng = np.random.default_rng(cfg.seed)
    out = {"train_images": [], "train_masks": [], "val_images": [], "val_masks": []}

    for i in range(cfg.n_train_big_tiles):
        img, msk = build_training_tile(annotation_crops, cfg,
                                       augment=(i >= cfg.n_train_big_tiles // 2),
                                       rng=rng)
        imgs, msks = cut_to_small_tiles(img, msk, cfg)
        out["train_images"] += imgs
        out["train_masks"] += msks

    for _ in range(cfg.n_val_big_tiles):
        img, msk = build_training_tile(annotation_crops, cfg, augment=False, rng=rng)
        imgs, msks = cut_to_small_tiles(img, msk, cfg)
        out["val_images"] += imgs
        out["val_masks"] += msks

    logger.info("dataset: %d train, %d val tiles (paper: 6480, 1620)",
                len(out["train_images"]), len(out["val_images"]))
    return out


def build_model(n_classes: int, cfg: SegmentationConfig | None = None):
    """DeepLab v3+ with a pretrained ResNet-50 backbone. [PAPER]

    torchvision ships deeplabv3_resnet50, which is v3 rather than v3+. The
    difference is the decoder: v3+ adds a decoder module that recovers spatial
    detail at object boundaries. For tissue-class segmentation at 2 um/pixel
    the practical difference is small, but it IS a deviation and belongs in the
    methods section. Use segmentation_models_pytorch for true v3+.
    """
    cfg = cfg or SegmentationConfig()
    try:
        import segmentation_models_pytorch as smp

        logger.info("using segmentation_models_pytorch DeepLabV3+ (matches paper)")
        return smp.DeepLabV3Plus(encoder_name="resnet50", encoder_weights="imagenet",
                                 classes=n_classes)
    except ImportError:
        from torchvision.models.segmentation import deeplabv3_resnet50

        logger.warning(
            "segmentation_models_pytorch not installed. Falling back to "
            "torchvision deeplabv3_resnet50, which is v3 not v3+. Record this "
            "deviation in the methods.")
        return deeplabv3_resnet50(weights_backbone="DEFAULT", num_classes=n_classes)


def check_acceptance(
    per_class_precision: dict[str, float], per_class_recall: dict[str, float],
    cfg: SegmentationConfig | None = None,
) -> tuple[bool, list[str]]:
    """CODA's acceptance criterion, applied as a criterion not a target. [PAPER]

    If any class falls below 90% precision or recall, the paper's instruction is
    to ADD ANNOTATIONS FOR THAT CLASS and retrain, repeating until the bar is
    met. It is not "train once and report whatever comes out".

    Returns (passed, list of failing classes).
    """
    cfg = cfg or SegmentationConfig()
    failing = [
        c for c in per_class_precision
        if per_class_precision[c] < cfg.acceptance_precision
        or per_class_recall.get(c, 0.0) < cfg.acceptance_recall
    ]
    if failing:
        logger.warning(
            "classes below the %.0f%% bar: %s. Per the protocol, add annotations "
            "for these classes to the training and testing images and retrain. "
            "Do not proceed to reconstruction with a failing model.",
            cfg.acceptance_precision * 100, failing)
    return (not failing), failing
