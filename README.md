# Handshake Asymmetry

**Learning the negotiation law of human–human handshakes, to correct the inherent asymmetry of human–robot handshakes.**

> Status: research idea + measured evidence (pre-implementation). No robot experiments yet.

---

## 1. The claim

Human–robot physical interaction has an **inherent asymmetry**: the human is real, the robot is not,
and almost every method silently asks the *robot* to absorb **all** of the mismatch in body size,
arm length, stiffness, perception and timing.

We argue the asymmetry is not the problem — **one-sided adaptation is**. Real humans shake hands
with partners up to 30 cm taller or shorter, and they do it by **splitting** the mismatch between
both bodies. That split is measurable, and it can be turned into a constraint for robot control.

We measured it from real human–human handshakes, then quantified what a humanoid would need.

---

## 2. Key measurements (all reproducible with the scripts in `scripts/`)

### 2.1 The handshake negotiation law (324 pairs / 58 dyads / 116 distinct people)

| height difference Δh | n | height ratio | clasp height | standoff | shorter partner's arm extension | lean (short / tall) |
|---|---:|---:|---:|---:|---:|---:|
| < −10 cm | 18 | 0.866 | **101.7 cm** | 88.2 cm | 84.9 % | 24.3° / 24.2° |
| −10 … −4 cm | 13 | 0.964 | 102.0 cm | 80.5 cm | 90.3 % | 20.9° / 28.2° |
| ±4 cm | 67 | 1.005 | 100.6 cm | 83.8 cm | 86.9 % | 23.8° / 25.3° |
| +4 … +10 cm | 30 | 1.052 | 97.6 cm | 83.8 cm | 87.1 % | 23.9° / 27.1° |
| > +10 cm | 61 | 1.156 | **95.0 cm** | 88.4 cm | 87.9 % | 22.6° / 27.6° |

Five regularities:

1. **The clasp height is an absolute social constant**, not a scaled quantity: it moves only
   **6.7 cm** while the height difference spans 30+ cm.
2. **Horizontally the clasp sits at the midpoint of the two bodies** (λ = 0.493 along the
   pelvis–pelvis line): horizontal adaptation is already symmetric.
3. **Standoff is 80–88 cm and grows when the mismatch is large** — people do *not* step closer
   to compensate; they stand slightly farther and reach up.
4. **Both partners lean their torso forward 22–28°, the taller one leans more.**
5. **Both extend to 84–90 % of their arm length** — nobody keeps a reserve.

### 2.2 What the asymmetry actually costs a humanoid

| | height | shoulder height | arm (shoulder→wrist) |
|---|---:|---:|---:|
| human (SMPL-X neutral) | 149.7 cm | 131.4 cm | **52.1 cm** |
| Unitree G1 (from `g1_29dof.xml`) | 132.0 cm | 108.5 cm | **38.7 cm** |
| difference | −17.7 cm | **−22.9 cm** | **−13.4 cm (74 % of human)** |

The arm ratio (74 %) is worse than the height ratio (88 %) — **the binding constraint is the
short arm, not the low stature.**

Required shoulder→clasp distance over 8,134 real contact frames, as a function of how much the
robot leans:

| robot lean | required (median) | p90 | exceeds arm | **reachable frames** |
|---|---:|---:|---:|---:|
| **0° (naive retargeting)** | 41.3 cm | 48.6 cm | 68 % | **32 %** |
| 15° | 36.0 cm | 43.3 cm | 31 % | 69 % |
| **25° (the lean humans actually use)** | 32.7 cm | 40.1 cm | 13 % | **87 %** |
| 35° | 29.8 cm | 37.5 cm | 7 % | 93 % |

**Torso lean alone takes reachability from 32 % to 87 %.** The remaining 13 % needs only a
median **3.2 cm concession** from the partner (p90 12.4 cm) — the same order of magnitude humans
already concede to each other.

### 2.3 Why this is not "a reaching problem"

We originally framed this as "the robot cannot reach". That framing missed the point:
a 0°-lean evaluation makes it look structurally impossible, but humans *always* lean.
The real statement is:

> The mismatch is absorbable **if and only if** it is split the way humans split it.
> One-sided absorption is what fails.

---

## 3. What we propose

Split the pipeline into three layers and give each layer its own share of the asymmetry:

| layer | what it does | what it contributes here |
|---|---|---|
| **Generation** | produce interaction references | learn the **concession prior** (vertical yield ratio ≈ 0.22, horizontal ≈ 0) |
| **Retargeting** | map human reference → robot morphology | solve the **negotiation target** (clasp height ≈ 100 cm, λ ≈ 0.5, robot lean 22–28°) |
| **Deployment** | closed loop with the real partner | execute **two-sided concession** (median 3.2 cm residual) |

Concretely: the retargeter should not only output a trajectory but also a **concession request**
("I can absorb 3.6 cm; please lower your hand / step in by 3.2 cm"), and the runtime layer should
negotiate that with the partner instead of silently stretching the robot to its limit.

This also replaces a hand-set hyper-parameter in prior work: *Beyond Mimicry* (PAIR) lets the
partner's upper body adapt through `w_hum`, but the weight is tuned by hand. Our measured law
supplies a data-driven value for it.

---

## 4. Repository map

```
docs/
  00-overview.md              one-page idea
  01-problem-and-framing.md   asymmetry, three layers, what prior work assumes
  02-handshake-geometry-law.md  the five laws, with the measurement protocol
  03-methods-survey.md        34 papers, method families, who-adapts-to-whom table, gap
  04-proposed-method.md       three knobs + concession law + experiment plan + risks
  05-data-and-audit.md        8 interaction datasets, 12-point quality audit, data pitfalls
scripts/
  smplx_fk.py                 SMPL-H loading + SMPL-X forward kinematics (Y-up caveat inside)
  law_handshake_geometry.py   measures the five regularities  -> law.json
  reachability_lean.py        measures reach vs lean + required concession -> reach.json
  audit_pairs.py              12-point quality audit for paired motion data -> audit.json
reports/
  report-data-verification.pdf      per-file verification of 5,675 motion files + dataset profiles
  report-methods-survey.pdf         the 34-paper methods survey
  analysis-three-layers.pdf         generation / retargeting / deployment attribution
```

---

## 5. Quickstart

Requirements: `numpy`, `torch`, `smplx`, and an `SMPLX_NEUTRAL.npz` model file (registration
required upstream). Input data is any two-person motion set in the SMPL-H 52-joint layout,
paired as `*_P1.npz` / `*_P2.npz` with a `raw_betas` field.

```bash
pip install numpy torch smplx

python scripts/law_handshake_geometry.py \
    --data-root /path/to/handshake_pairs \
    --model /path/to/SMPLX_NEUTRAL.npz \
    --out law.json

python scripts/reachability_lean.py \
    --data-root /path/to/handshake_pairs \
    --model /path/to/SMPLX_NEUTRAL.npz \
    --out reach.json

python scripts/audit_pairs.py \
    --root /path/to/handshake_pairs \
    --model /path/to/SMPLX_NEUTRAL.npz \
    --out audit.json --workers 8
```

Data used for the numbers in this repository: the `Handshake` (A001) subset of **Inter-X**
(324 pairs). `CHI3D` (`Handshake`) and `Hi4D` are kept as independent cross-validation sets.

---

## 6. Two data pitfalls you must know before trusting any number

1. **Released two-person data has `betas` all zeros.** Every subject is fitted to the same
   neutral skeleton: the P1/P2 forearm-length ratio is exactly `1.000000` in all 714 released
   pairs, so the "body-size mismatch" does not exist in the data as shipped. The real body is
   only in `raw_betas`; re-running FK with `raw_betas` restores a 32 cm height range while
   changing contact geometry by only +0.6 cm median.
2. **The floor is only aligned for the lower of the two partners.** One partner floats by a
   median 6.4 cm (62 % of pairs > 5 cm). Align each person by their own lowest foot.

Details and the full audit are in `docs/05-data-and-audit.md`.

---

## 7. Relationship to prior work

24 human–robot interaction papers were surveyed. Distribution of *who adapts*:

| how adaptation is allocated | papers |
|---|---:|
| robot absorbs everything, partner fixed | **14** |
| partner predicted/modelled but not asked to move | 4 |
| partner may adapt — but the weight is hand-tuned | **1** (PAIR / `w_hum`) |
| explicit role assignment (task level, not contact level) | 2 |
| mechanical compliance absorbs the difference | 1 |

Only **one** paper spans all three layers (generation + retargeting + deployment). The
layer-to-layer seams are exactly where our contribution sits. See `docs/03-methods-survey.md`.

---

## 8. License

MIT — see `LICENSE`. The motion data itself keeps its upstream licenses and is **not** included
in this repository.
