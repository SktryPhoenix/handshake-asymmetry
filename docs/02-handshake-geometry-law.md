# The handshake negotiation law

Reproduce with `scripts/law_handshake_geometry.py`.

## Data and protocol

- **Inter-X, action class A001 (`Handshake`)**: 324 pairs, 58 dyads, 116 distinct individuals,
  56,134 frames. Each dyad keeps a fixed pair of subjects, so splitting by dyad keeps subjects
  disjoint.
- Body size from **`raw_betas`**, not `betas` (which is all zeros — see `05-data-and-audit.md`).
- **Every person is grounded by their own lowest foot.** The released data only aligns the pair
  jointly and leaves one partner floating by ~6.4 cm.
- Contact = min palm-to-palm distance < 8 cm over all four hand combinations; 207/324 pairs
  reach contact. The clasp is the midpoint of the two palms over contact frames.
- Quantities measured per pair: clasp height, clasp position along the pelvis–pelvis line (λ),
  pelvis distance (standoff), wrist–shoulder distance divided by arm length (extension), and the
  angle between (shoulder − pelvis) and the vertical (lean).

## The five regularities

### L1 — the clasp height is an absolute social constant

| Δh | n | h₂/h₁ | clasp height | clasp / shorter shoulder |
|---|---:|---:|---:|---:|
| < −10 cm | 18 | 0.866 | 101.7 cm | 0.854 |
| −10…−4 | 13 | 0.964 | 102.0 cm | 0.762 |
| ±4 | 67 | 1.005 | 100.6 cm | 0.774 |
| +4…+10 | 30 | 1.052 | 97.6 cm | 0.784 |
| > +10 | 61 | 1.156 | 95.0 cm | 0.818 |

Across a 30+ cm height gap the clasp moves 6.7 cm. It is **not** a fraction of either body and
**not** the midpoint in height. Design consequence: target an absolute height near 1.0 m.

### L2 — horizontally the split is already symmetric

λ (0 = partner 1, 0.5 = midpoint) = **0.493** median, p10 0.414, p90 0.576.
No partner claims a positional advantage; horizontal adaptation is naturally balanced.

### L3 — standoff grows with the mismatch

| Δh | standoff (pelvis distance) |
|---|---:|
| < −10 cm | 88.2 cm |
| −10…−4 | 80.5 cm |
| ±4 | 83.8 cm |
| +4…+10 | 83.8 cm |
| > +10 | 88.4 cm |

Counter-intuitive but consistent: when the mismatch is large, people stand **slightly farther
apart** and reach up, rather than closing in. Any controller that closes distance to gain reach is
deviating from the human pattern.

### L4 — both partners lean, the taller one more

| Δh | shorter's lean | taller's lean |
|---|---:|---:|
| < −10 cm | 24.3° | 24.2° |
| −10…−4 | 20.9° | **28.2°** |
| ±4 | 23.8° | 25.3° |
| +4…+10 | 23.9° | 27.1° |
| > +10 | 22.6° | **27.6°** |

### L5 — both extend to 84–90 % of their arm

Nobody keeps a reserve. A "comfortable" controller that stops at 70 % extension is not mimicking
human handshakes.

## Derived: the concession law

Fitting the vertical trend gives a **yield ratio ≈ 0.22**: the clasp drops ~6.7 cm while the
height gap spans ~30 cm. Horizontally the ratio is ≈ 0.

This ratio is what should replace a hand-tuned partner-adaptation weight in retargeting
formulations that allow the partner to move (e.g. PAIR's `w_hum`).

## Sanity expectations

For a pair with Δh in the largest bucket, expect: clasp ≈ 95–103 cm, standoff ≈ 84–88 cm,
λ ≈ 0.5, extension ≈ 84–90 %, lean ≈ 22–28°. If your pipeline produces a clasp at 70 cm or a
standoff of 50 cm, it is not producing a handshake.

## Known limits of this measurement

- All dyads are adults; the robot-vs-human ratio (0.882) sits at the ~4th percentile of observed
  dyad ratios (0.734–1.477), i.e. slightly beyond the bulk of the human data. Extrapolation is
  mild but must be stated.
- A001 contains 55 % contact-positive pairs; 36 % never come within 10 cm at all (labelled
  handshake but visually not one). Buckets therefore are not uniformly populated.
