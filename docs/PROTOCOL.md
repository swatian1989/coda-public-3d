# Protocol: applying the CODA pipeline to public data

Two arms, chosen after establishing what each dataset can actually support.
Feasibility was verified against the data, not assumed from the papers.

Reference: Kiemen A et al. CODA: quantitative 3D reconstruction of large tissues
at cellular resolution. *Nat Methods* 2022;19:1490-1499. PMID 36280719.

---

## Feasibility, established before any analysis

### TCGA cannot support the three-dimensional stages. Verified.

Queried the GDC API directly for TCGA-BRCA slide files (n = 3,112):

| slides per patient | patients |
|---|---|
| 1 | 230 |
| 2 | 60 |
| 3 | 13 |
| 4 | 1 |
| 7 (maximum) | 1 |

The `section_location` field returns TOP (344) and BOTTOM (123). Those labels
exist because slides are taken from opposite ends of a block to sample different
regions, which is the opposite of consecutive. Slide types are DX (diagnostic
FFPE), TS and BS (frozen top and bottom) and MS, so a patient's slides differ in
block, in preparation, and in region.

CODA used 4,114 sections across 13 samples, cut every 4 um, about 316 sections
per sample. Stages 1, 2, 5 and 6, namely registration, registration quality
control, reconstruction and volumetric quantification, are therefore impossible
on TCGA for any cancer type. No processing converts seven non-consecutive slides
into a volume.

### CODA's own data is not public

The Data Availability statement reads "Data is available upon request from the
corresponding author." Only the MATLAB code is released, at
github.com/ashleylk/CODA. The pancreas series cannot be reanalysed directly.

### Consequence

Arm 1 uses the only openly licensed serial series deep enough for the full
pipeline. Arm 2 uses TCGA for the stages it genuinely supports and says so.

---

## Arm 1: Kartasalo prostate, 260 consecutive sections

Etsin `c76335fa-cdcf-4ddc-ab1c-1882bad82861`, CC BY 4.0, access type Open.
260 sections at 5 um is a block 1.3 mm deep, 5.5 times the 47-section liver
series analysed previously, which matters for one specific measurement (below).

| Stage | Runs | Note |
|---|---|---|
| 1 registration | yes | two-scale rigid with the corrected rotation estimator |
| 2 registration QC | yes | TRE and ATRE against 259 pairs of operator landmarks |
| 3 cell detection | partial | two automatic detectors; no human annotation exists |
| 4 segmentation | no | needs annotated tiles and a GPU |
| 5 reconstruction | yes | 260 sections into a volume |
| 6 connectivity | yes | 2D versus 3D object counting |
| 7 fibre alignment | yes | and see below |

**Why the depth matters.** On the liver series the sectioning-angle comparison
failed: the block was only 235 um deep, so orthogonal planes were thin slabs and
a shuffle control showed the anisotropy index was reading slab geometry rather
than tissue. At 1.3 mm the prostate block is thick enough for that comparison to
be meaningful. It will be run with the same shuffle control, and reported only
if the control passes.

### Two landmark traps specific to this series, both verified

**The landmarks are pairwise, not through-stack.** The table has 259 rows for
260 sections and sixteen coordinate columns: Y1/X1 is a point on section n and
Y2/X2 is that same point on section n+1. A landmark exists only for the pair it
was drawn on. Accumulated error therefore cannot be the residual about a line
fitted down z, which is the liver definition and the one implemented in
`coda_my.qc.accumulated_tre`. It must be the cumulative resultant of the mean
pairwise displacement vectors, which is implemented separately in
`loaders/kartasalo_prostate.py`. Using the liver form here would silently
produce a number that answers a different question.

**The two observers are not repeated measurements.** For the liver both
observers annotated the same four laser-cut holes, so their disagreement was an
annotation noise floor, 6.8 um median, and was reported as one. The prostate has
no holes and each observer chose their own anatomical features: observer 1's
point k lies a median 5,750 um from observer 2's point k, and 1,286 um from the
nearest of observer 2's points, so they are not the same features in a different
order. Inter-observer distance is not a noise floor here and will not be quoted
as one. One observer's landmarks are used throughout and named.

### Baselines measured before registration

Unregistered, observer 1, 259 pairs:

- pairwise TRE mean 489.2 um, median 310.4, maximum 2,392.3
- accumulated TRE, cumulative vector form, mean 776.7 um, maximum 2,043.3

Any registration that does not beat 489 um is doing harm. This is the same test
the liver arm applied, where the stock pipeline failed it.

### Cost

The archive is a single 63.79 GB zip and the download service ignores HTTP range
requests, so the transfer cannot resume and cannot be fetched in part. The
prostate series sits after the liver in the archive, so the whole stream is
required, roughly 4.6 hours. The zip is parsed as it arrives and only prostate
members are written, so peak disk is the series, not the archive.

---

## Arm 2: TCGA-BRCA, the stages it supports

Diagnostic (DX) slides only. Frozen TS and BS slides carry freezing artefact
that changes texture and would confound any morphometric comparison.

| Stage | Runs | Note |
|---|---|---|
| 1, 2, 5, 6 | no | no serial sections; verified above |
| 3 cell detection | yes | at cohort scale, on human breast |
| 4 segmentation | no | needs annotated tiles and a GPU |
| 7 fibre alignment | yes | H&E has an eosin channel |
| stereology | yes | volume fraction and corrected volumetric density |

**What this arm adds that Arm 1 cannot.** Arm 1 is mouse prostate. Arm 2 is
human breast at cohort scale, which is the clinical question, and it allows the
sectioning-angle caveat to be handled honestly: on single sections the cutting
angle is unknown, so fibre alignment is reported as a distribution across many
slides and never as a per-patient property.

**What it cannot establish.** No volume, no connectivity, no overcounting ratio,
and no validation of cell detection against human annotation, because none
exists for these slides.

---

## Reporting rules for both arms

- Every figure and table states REAL with dataset and n, or SIMULATED.
- A stage that did not run is named with its missing input. A reader must never
  be able to mistake an absent result for a null one.
- Method reproduction and biological reproduction are distinguished explicitly.
  Mouse prostate cannot reproduce a human pancreas finding, and no attempt is
  made to claim otherwise.
- Any measurement that depends on an assumption reports the assumption and,
  where two assumptions oppose each other, reports both rather than folding them
  into one number.
