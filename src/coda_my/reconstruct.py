"""CODA 3D reconstruction and quantification from a registered stack.

Implements the Online Methods:

  - Stack labelled 2D sections into a 3D matrix using the registration result.
  - Resample from 2 x 2 x 12 um voxels to an isotropic 12 x 12 x 12 um grid.
    Every downstream volume statistic assumes isotropy; skipping this silently
    weights the z axis six times too heavily.
  - Tissue composition = voxels per class / total tissue voxels.
  - 2D to 3D cell count extrapolation with the nuclear diameter correction.
  - Connectivity labelling (MATLAB bwlabeln equivalent) to count spatially
    independent lesions in 3D rather than per section.
  - z-projections per tissue class.

The headline CODA result rests on the connectivity step: counting lesions on
2D sections overcounted the true 3D number by an average of 12.3-fold and up to
40-fold, because lesions that look separate in one plane are connected in
another. That comparison is reproduced by `overcounting_ratio`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import ndimage

logger = logging.getLogger(__name__)

# [PAPER] In-situ measured nuclear diameters, Extended Data Fig 2b, in microns.
# These are pancreas values. MEASURE YOUR OWN for a different tissue: the
# correction scales the cell count directly, so borrowed diameters bias every
# density you report.
NUCLEAR_DIAMETER_UM = {
    "ductal_epithelium": 4.2, "panin": 5.5, "pdac": 6.7, "smooth_muscle": 4.1,
    "islets": 3.4, "acini": 4.0, "ecm": 2.5, "fat": 70.0,
}


@dataclass
class ReconstructionConfig:
    section_thickness_um: float = 4.0     # [PAPER] T, cut every 4 um
    sections_skipped: int = 3             # [PAPER] every 3rd section stained
    xy_mpp: float = 2.0                   # [PAPER] segmentation resolution
    isotropic_voxel_um: float = 12.0      # [PAPER] 12 x 12 x 12 um
    nuclear_diameters: dict = field(default_factory=lambda: dict(NUCLEAR_DIAMETER_UM))
    connectivity: int = 3                 # 26-neighbour in 3D, matches bwlabeln

    @property
    def axial_spacing_um(self) -> float:
        return self.section_thickness_um * self.sections_skipped


def stack_to_volume(
    labelled_sections: list[np.ndarray], cfg: ReconstructionConfig | None = None
) -> np.ndarray:
    """Stack registered label images into an ISOTROPIC 3D label matrix. [PAPER]

    Input sections must already be registered and must all share a shape.
    Nearest-neighbour resampling is used throughout because these are class
    labels, not intensities; linear interpolation would invent classes that do
    not exist between, say, class 2 and class 4.
    """
    cfg = cfg or ReconstructionConfig()
    shapes = {s.shape for s in labelled_sections}
    if len(shapes) != 1:
        raise ValueError(f"sections have differing shapes: {shapes}. Register first.")

    vol = np.stack(labelled_sections, axis=0).astype(np.int16)   # (z, y, x)

    zoom_xy = cfg.xy_mpp / cfg.isotropic_voxel_um
    zoom_z = cfg.axial_spacing_um / cfg.isotropic_voxel_um
    iso = ndimage.zoom(vol, (zoom_z, zoom_xy, zoom_xy), order=0)

    logger.info("volume %s at %.0f um isotropic (%.3f mm^3)", iso.shape,
                cfg.isotropic_voxel_um, volume_mm3(iso, cfg))
    return iso


def volume_mm3(volume: np.ndarray, cfg: ReconstructionConfig,
               label: int | None = None) -> float:
    """Volume in mm^3. 1 voxel = 12^3 x 1e-9 mm^3 at the paper's resolution."""
    voxel_mm3 = (cfg.isotropic_voxel_um ** 3) * 1e-9
    n = int((volume > 0).sum()) if label is None else int((volume == label).sum())
    return float(n * voxel_mm3)


def tissue_composition(
    volume: np.ndarray, class_names: dict[int, str],
    cfg: ReconstructionConfig | None = None,
) -> pd.DataFrame:
    """Fraction of tissue volume per class. [PAPER]

    Denominator is TISSUE voxels, not all voxels. Including background would
    make every composition a function of how much empty slide was scanned.
    """
    cfg = cfg or ReconstructionConfig()
    tissue = int((volume > 0).sum())
    if tissue == 0:
        raise ValueError("no tissue voxels; check the label convention (0 = background)")

    rows = []
    for label, name in class_names.items():
        if label == 0:
            continue
        n = int((volume == label).sum())
        rows.append({"label": label, "class": name, "n_voxels": n,
                     "volume_mm3": volume_mm3(volume, cfg, label),
                     "fraction": n / tissue})
    return pd.DataFrame(rows).sort_values("fraction", ascending=False)


def extrapolate_3d_cell_count(
    counts_2d: dict[str, int], cfg: ReconstructionConfig | None = None
) -> dict[str, float]:
    """2D to 3D cell count with the nuclear-diameter correction. [PAPER]

        C3D = sum over images, subtypes of  C_image * 3T / (T + D_subtype)

    The logic: a nucleus is detected if ANY part of it intersects the section,
    so the effective sampling thickness is the section thickness PLUS the
    nuclear diameter, not the section thickness alone. Without this correction
    cell counts are inflated, and inflated most for the largest nuclei, which
    biases comparisons between cell types systematically rather than randomly.

    The factor 3 is because two of every three sections were skipped.
    """
    cfg = cfg or ReconstructionConfig()
    T = cfg.section_thickness_um
    out = {}
    for subtype, c in counts_2d.items():
        d = cfg.nuclear_diameters.get(subtype)
        if d is None:
            logger.warning("no nuclear diameter for '%s'; measure it, do not guess. "
                           "Skipping the correction for this class.", subtype)
            out[subtype] = float(c * cfg.sections_skipped)
            continue
        out[subtype] = float(c * cfg.sections_skipped * T / (T + d))
    return out


def label_connected_objects(
    volume: np.ndarray, target_label: int,
    cfg: ReconstructionConfig | None = None,
) -> tuple[np.ndarray, int]:
    """3D connected-component labelling. MATLAB bwlabeln equivalent. [PAPER]"""
    cfg = cfg or ReconstructionConfig()
    structure = ndimage.generate_binary_structure(3, cfg.connectivity)
    labels, n = ndimage.label(volume == target_label, structure=structure)
    logger.info("%d spatially independent objects for label %d", n, target_label)
    return labels, n


def overcounting_ratio(
    volume: np.ndarray, target_label: int,
    cfg: ReconstructionConfig | None = None,
) -> pd.DataFrame:
    """Reproduce the central CODA result: 2D lesion count over 3D truth. [PAPER]

    For each section containing the target class, counts objects distinct in 2D
    and divides by the number of distinct 3D objects appearing on that section.
    CODA reported an average 12.3-fold overcount, up to 40-fold, p < 1e-5.

    Returns per-section ratios plus their mean and SD, so the result can be
    compared against the published figure directly.
    """
    cfg = cfg or ReconstructionConfig()
    labels3d, n3d = label_connected_objects(volume, target_label, cfg)
    if n3d == 0:
        return pd.DataFrame(columns=["z", "n_2d", "n_3d", "ratio"])

    struct2d = ndimage.generate_binary_structure(2, 2)
    rows = []
    for z in range(volume.shape[0]):
        plane = volume[z] == target_label
        if not plane.any():
            continue
        _, n2d = ndimage.label(plane, structure=struct2d)
        present = np.unique(labels3d[z][plane])
        n_on_plane = int(len(present[present > 0]))
        if n_on_plane:
            rows.append({"z": z, "n_2d": n2d, "n_3d": n_on_plane,
                         "ratio": n2d / n_on_plane})

    df = pd.DataFrame(rows)
    if not df.empty:
        logger.info("2D/3D overcounting: mean %.1f-fold, max %.1f-fold "
                    "(CODA reported mean 12.3, max 40)",
                    df["ratio"].mean(), df["ratio"].max())
    return df


def object_metrics(
    volume: np.ndarray, target_label: int,
    cfg: ReconstructionConfig | None = None,
) -> pd.DataFrame:
    """Per-object volume, primary axis length and bounding box. [PAPER]

    CODA used these to classify pancreatic precursors into tubular, dilated and
    lobular 3D phenotypes. The equivalent question in another tissue is whether
    lesions of the same 2D appearance separate into distinct 3D morphologies.
    """
    cfg = cfg or ReconstructionConfig()
    labels3d, n = label_connected_objects(volume, target_label, cfg)
    if n == 0:
        return pd.DataFrame()

    voxel_mm3 = (cfg.isotropic_voxel_um ** 3) * 1e-9
    rows = []
    for obj_id, sl in enumerate(ndimage.find_objects(labels3d), start=1):
        mask = labels3d[sl] == obj_id
        n_vox = int(mask.sum())
        extent = np.array([s.stop - s.start for s in sl]) * cfg.isotropic_voxel_um
        rows.append({
            "object_id": obj_id,
            "n_voxels": n_vox,
            "volume_mm3": n_vox * voxel_mm3,
            "primary_axis_um": float(extent.max()),
            "extent_z_um": float(extent[0]),
            "extent_y_um": float(extent[1]),
            "extent_x_um": float(extent[2]),
            "elongation": float(extent.max() / max(extent.min(), 1e-9)),
            "n_sections_spanned": int(extent[0] / cfg.isotropic_voxel_um),
        })
    return pd.DataFrame(rows).sort_values("volume_mm3", ascending=False)


def z_projection(volume: np.ndarray, target_label: int) -> np.ndarray:
    """Sum a class along z and normalise. [PAPER] Extended Data Figs 6-8."""
    proj = (volume == target_label).sum(axis=0).astype(float)
    m = proj.max()
    return proj / m if m > 0 else proj
