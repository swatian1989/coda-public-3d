# coda-public-3d

The CODA 3D histology pipeline applied to openly available data. Two arms,
chosen after establishing what each dataset can actually support.

## Feasibility, established before any analysis

**TCGA cannot support the three-dimensional stages.** Querying the GDC API for
TCGA-BRCA returned 3,112 slide files with a maximum of **seven slides for any
one patient**, and the `section_location` field reports TOP and BOTTOM because
slides are deliberately taken from opposite ends of a block. There are no
consecutive sections in TCGA for any cancer type, so stages 1, 2, 5 and 6 are
impossible there and are not attempted.

**The source publication's data is not public**, being available only on request
from its authors. Arm 1 therefore uses the openly licensed serial series that is
deep enough for the full pipeline.

## Arm 1: mouse prostate, 260 consecutive sections, 1.3 mm block

| metric | no transform | registered |
|---|---|---|
| pairwise TRE | 489.2 um | **65.5 um** |
| accumulated TRE | 776.7 um | 342.2 um |
| pixel correlation | | 0.989 median, 0 of 260 flagged |

Two-dimensional counting overestimates object number **4.37-fold** across all
objects and **41-fold** restricted to structures above 10^5 um^3. The source
reported 12.3-fold on average and up to 40-fold. The ratio is a property of what
is counted and at what size, so it is reported as a curve rather than a number.

Composition error stays near the 5 percent tolerance out to 15 um of section
spacing, approximately reproducing the published claim of under 5 percent to
12 um.

**Read the accumulated error honestly.** It is near zero at the centre reference
and rises to 600 to 850 um at both ends of the stack, approaching the 777 um
do-nothing baseline. Pairwise alignment is excellent and the reconstruction is
reliable locally, but over the full 1.3 mm the stack bends. Anything measured
across the whole block inherits that drift; anything local does not. This is the
failure the cumulative-vector definition of accumulated error exists to expose.

## Arm 2: TCGA-BRCA, the stages it supports

Ten diagnostic slides, drawn at random from the 151 cases that carry BCSS tissue
annotations so that segmentation has ground truth on the same slides it is
applied to. Nuclear density median 5,148 per mm^2; fibre anisotropy median 0.354
over 1,209 windows.

Neither is validated against human annotation, because none exists for these
slides, so the published 90 percent precision and recall gate is not tested and
is not claimed. Anisotropy is reported as a distribution and never as a
per-patient value, because the cutting angle of a single section is unknown and
cannot be corrected.

## Stage 4, segmentation

The one stage needing a GPU. Open the notebook in Colab, select a T4 runtime and
run all cells; it trains DeepLab v3+ on BCSS annotations, reports a confusion
matrix with per-class precision and recall against the 90 percent gate, and
exports weights for CPU inference here.

## Running it

```bash
pip install -r requirements.txt
python scripts/fetch_kartasalo_prostate.py   # 63.79 GB stream, unresumable
python scripts/run_prostate_pipeline.py      # stages 1, 2, 5, 6, 7
python scripts/fetch_tcga_brca.py --n 10     # 9.6 GB
python scripts/run_tcga_analysis.py          # stages 3, 7, stereology
python scripts/run_report.py
```

## Caveats

- Arm 1 is **mouse prostate**. It establishes that the method runs and how
  accurately. It is not a breast finding.
- **No three-dimensional reconstruction of breast tissue** appears here, and none
  is possible without serial breast sections, which no public dataset provides.
- Objects counted in stage 6 are glandular lumina separated by an intensity band,
  not the ten classes of a trained segmentation.

## Data

Code only. No images and no generated results; everything is reproduced by
running the pipeline. `data/reference/` holds values transcribed from published
tables, which are inputs rather than outputs.

Kiemen A et al. CODA. *Nat Methods* 2022;19:1490-1499. PMID 36280719
Kartasalo K et al. *Bioinformatics* 2018;34:3013-3021. PMID 29684099
Amgad M et al. *Bioinformatics* 2019;35:3461-3467. PMID 30726865
