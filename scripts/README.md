# scripts

All scripts are standalone and take paths as arguments — nothing is hard-coded.

| file | purpose | output |
|---|---|---|
| `smplx_fk.py` | motion loading + SMPL-X forward kinematics; **also the joint index table** | module |
| `law_handshake_geometry.py` | measures the five negotiation regularities | `law.json` + table |
| `reachability_lean.py` | required reach vs robot lean angle; residual concession | `reach.json` + table |
| `audit_pairs.py` | 12-point quality audit for paired motion data | `audit.json` |

## Requirements

```bash
pip install numpy torch smplx
```

plus an `SMPLX_NEUTRAL.npz` model (upstream registration required).

## Input layout

````
<data-root>/
  SEQ_A_P1.npz      # poses (T,156), trans (T,3), betas (16,), raw_betas (16,)
  SEQ_A_P2.npz
  ...
````

`raw_betas` is **required** by `law_handshake_geometry.py` and `reachability_lean.py`: the released
`betas` field is all zeros, so without `raw_betas` every subject is the same body and no body-size
mismatch exists to measure.

`audit_pairs.py` additionally accepts the mirror layout `.../smplh_52_2p/P1.npz` and skips `*_1p`
paths (their P1/P2 live in separate coordinate frames).

## Two conventions that silently break results

1. **FK output is Y-up.** Height is component 1. Component 2 is depth. Using 2 gives meaningless
   "palm heights" (we measured 2 cm before noticing).
2. **Ground each person by their own lowest foot.** The released pair alignment leaves one partner
   floating by ~6 cm.

## Runtime notes

- `smplx.create` reloads a ~100 MB model file on every call; `smplx_fk.py` caches it per process.
- On shared CPU boxes set `OMP_NUM_THREADS=1`; otherwise torch spawns enough threads to thrash and
  a 2-minute audit takes tens of minutes.
