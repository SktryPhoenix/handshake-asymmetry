# Data used, and the 12-point audit that made it usable

Reproduce with `analysis/audit_pairs.py`. This documents why every number in this repository uses
`raw_betas` and per-person floor alignment.

## 1. Datasets examined

Eight motion sets in a unified SMPL-H 52-joint layout were downloaded and raked over
(5,675 files, 1,472,202 frames, 13.64 h, ~1.5 GB):

| dataset | files | frames | two-person pairs | type | `raw_betas` | objects |
|---|---:|---:|---:|---|---|---|
| **Inter-X A001** | 648 | 112,268 | **324** | handshake only | yes | no |
| **CHI3D** | 381 | 68,823 | 51 (incl. **11 Handshake**) | grab / handshake / hit | yes | no |
| **Hi4D** | 400 | 47,780 | 100 | 23 action classes, 1 handshake | yes | no |
| TRUMANS | 999 | 204,228 | — | single person + scene | no | no |
| HumanSC3D | 688 | 120,978 | — | single person + scene | no | no |
| Fit3D | 944 | 338,904 | — | single person, 173 exercises | no | no |
| GRAB | 990 | 303,254 | — | hand–object actions | no | **no** |
| EgoBody | 625 | 275,967 | — | single person, social scenes | no | no |

Handshake material available: **324 + 11 + 1 = 336 pairs from two independent capture setups**,
which allows cross-validation of the law (Inter-X clasp ≈ 88.5 cm, CHI3D ≈ 95.4 cm).

Only the two-person sets carry `raw_betas`; **all eight sets have no object field**, so human–object
interaction is out of scope without the official releases (BEHAVE / GRAB official / OMOMO / ARCTIC).

## 2. Per-file verification

- loadable: 5,674 / 5,675 (one zero-byte file in GRAB, re-downloaded)
- required keys (`poses`, `trans`): complete
- `num_frames` vs array length: consistent everywhere
- NaN / Inf: 0
- `poses` dimension: 156 (52 joints × 3) everywhere
- frame rate: 30 fps everywhere
- `betas` all-zero: **100 % in all eight sets**
- `raw_betas`: present in 100 % of two-person files, absent in all single-person files

## 3. The 12 audit metrics

| # | metric | what it can falsify |
|---|---|---|
| 1 | frames / fps | is the clip long enough |
| 2 | P1/P2 limb ratio at zero betas | **body-size difference exists at all** |
| 3 | P1/P2 limb ratio with `raw_betas` | the true size difference |
| 4 | rest height per subject | how much height variation was destroyed |
| 5 | palm-to-palm minimum | is there contact |
| 6 | hand-pair distribution | can a single hand be hard-coded |
| 7 | contact-frame ratio | is contact an event or a state |
| 8 | palm height at contact | handshake (~90 cm) vs hug / dance |
| 9 | lowest joint height | global floor offset |
| 10 | `conversion_y_shift` consistency | can the metadata be trusted |
| 11 | palm acceleration jitter | reconstruction quality |
| 12 | foot slide per frame | are the feet sticky |

## 4. The three findings that changed the project

### 4.1 `betas` is all zeros — the body-size mismatch does not exist in the data as shipped

Forearm-length ratio between the two partners:

| | median | min | max |
|---|---:|---:|---:|
| zero betas (as released) | **1.000000** | 1.000000 | 1.000000 |
| `raw_betas` (true bodies) | ~1.0 | 0.883 | 1.149 |

Across 475 released pairs the zero-beta ratio is exactly 1 to 1e-6: every subject is the same
neutral body. Real heights, recovered from `raw_betas`, span **136.7–168.9 cm** (range 32.2 cm);
33 % of pairs differ by more than 10 cm, median pair difference +1.3 cm.

**Recovery test:** re-running FK with `raw_betas` and aligning each person's own floor changes the
palm-to-palm minimum by only **+0.6 cm median** (85 % of pairs within 5 cm; 10 % change >10 cm).
So the data is usable — it was just read through the wrong field.

### 4.2 The floor is aligned for the lower partner only

Lowest-joint height is −14.5 to −15.0 cm in all three two-person sets, and
`conversion_y_shift` matches it in only 75.9 % of clips. Comparing each partner's own lowest foot:
the taller-standing partner floats by a **median 6.4 cm** (62 % of pairs > 5 cm, 23 % > 10 cm).

If a retargeter enforces "feet on the ground" on the joint pair alignment, it drags the floating
partner down and the whole upper body with it. Align each person independently.

### 4.3 Contact is an event, not a state

| | Inter-X A001 | CHI3D | Hi4D |
|---|---:|---:|---:|
| contact-frame ratio | 19.0 % | 5.2 % | 5.4 % |
| pairs with ≥5 % contact frames | 178 / 324 | 12 / 51 | 19 / 100 |
| palm height at contact | **88.5 cm** | **89.7 cm** (Handshake-only 95.4) | 120.7 cm (mostly hugs) |

36 % of A001 clips never come within 10 cm; 4 clips never within 50 cm. A hard contact constraint
is therefore wrong by construction, and any metric must be computed over contact windows rather
than over the whole clip.

## 5. Additional trap: `1p` and `2p` variants

Data mirrors ship two variants. The `1p` files are each fitted in their **own** coordinate frame;
pairing them is meaningless (one measured pair showed an 8 cm pelvis separation, versus 71 cm in
the `2p` variant). `audit_pairs.py` therefore skips any file under a `*_1p` path.

## 6. Audit snapshot (corrected)

| metric | Inter-X A001 | CHI3D | Hi4D |
|---|---:|---:|---:|
| pairs | 324 | 51 | 100 |
| palm minimum (median) | 8.40 cm | 15.19 cm | 16.56 cm |
| — handshake subset only | 8.40 | **2.22** (11 pairs) | — |
| contact-frame ratio | 19.0 % | 5.2 % | 5.4 % |
| palm height at contact | 88.5 cm | 89.7 cm | 120.7 cm |
| lowest joint | −14.8 cm | −14.4 cm | −14.9 cm |
| cross-hand share | 43 % | 18 % | 38 % |
