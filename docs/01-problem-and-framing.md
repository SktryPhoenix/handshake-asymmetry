# Problem and framing

## 1. Two framings of the same failure

A handshake between a short-armed robot and a taller human has an obvious kinematic symptom:
the robot's arm cannot reach the clasp. It is tempting to treat this as a **reaching problem** —
add degrees of freedom, extend the arm, relax joint limits.

We argue that is the wrong framing, for one measured reason: **humans solve the same mismatch
without a strong reach deficit**, because they distribute it. The very same contact frames that
are unreachable at 0° torso lean become 87 % reachable once the robot leans by the amount humans
actually lean (25°). The reach deficit is real but small (median 3.6 cm); what makes the naive
solution fail is the **allocation**, not the deficit.

## 2. Where the asymmetry lives

| axis | human–human | human–robot | can be learned from motion data? |
|---|---|---|---|
| geometric | height / arm / shoulder height differ, both yield | robot shorter, arm proportionally shorter | **yes** — the five laws in `02-...` |
| mechanical | both bodies are compliant | robot rigid, little force sensing | no (needs simulation / hardware) |
| perceptual | both feel grip and slip | robot blind to most of it | no (no force channel in the data) |
| temporal | mutual approach timing, hold duration | latency on the robot side | partially |
| social | grip strength, duration, gaze | no expression, no skin | no |

So the data can support the **geometric and temporal** axes; mechanical and social axes are out
of scope here and must be flagged as such in any claim.

## 3. Three layers, and what each one currently assumes

| layer | input → output | current assumption about asymmetry |
|---|---|---|
| **generation** | demos / video → usable interaction reference | references assume **symmetric** partners: same body, both reactive, neither needs to yield |
| **retargeting** | human reference + robot morphology → executable trajectory | the **robot absorbs everything**, the partner is fixed |
| **deployment** | trajectory + online partner → real interaction | predict the partner; adjust the robot; **never ask the partner to yield** |

Evidence for the middle row: of 24 surveyed human–robot interaction papers, 14 resolve the
mismatch purely on the robot side, and the single exception lets the partner adapt through a
**hand-tuned** weight (PAIR's `w_hum`).

## 4. Consequence for the design

Because concession needs a partner *in the loop*, the actual correction can only happen at
deployment. But its **prior** must come from generation (measured from real handshakes), and its
**target** must come from retargeting (a solved clasp pose, not a reach pose). That is the whole
argument for splitting the contribution across the three layers instead of adding another
retargeting loss term.

## 5. Open questions we do not claim to have solved

1. Does a 3.2 cm concession feel natural to a human, or merely become feasible? (needs a user study)
2. How should a concession request be *communicated*? (speech? a small guided motion? impedance?)
3. What happens when the partner does not comply — is the fallback a re-approach?
4. Mechanical compliance (SEA, impedance) can absorb more than geometry; where is the boundary
   between what geometry and what compliance should absorb?
