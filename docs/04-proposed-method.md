# Proposed method: a negotiation target plus a concession budget

Status: design, not implemented. This document is the specification we intend to test.

## 1. Three knobs taken from the measured law

| knob | value from data | who executes it |
|---|---|---|
| clasp height | **absolute**, ≈ 1.00 m (96–103 cm across the height range) | joint target for both |
| clasp horizontal position | **midpoint** of the two pelvises (λ ≈ 0.49) | already symmetric |
| standoff | 80–88 cm, growing with the mismatch | the robot walks into it; the partner does not step in |
| robot torso lean | **22–28°** | robot |
| extension budget | 84–90 % of arm length | constraint on both |

## 2. Why these knobs close the gap

Measured over 8,134 real contact frames: required shoulder→clasp distance drops from
41.3 cm (median) at 0° lean to 32.7 cm at 25° lean, versus a 38.7 cm arm. Reachable frames go
from 32 % to 87 %. The remaining 13 % needs either 2.7 cm of step-in or 9.7 cm of clasp lowering
(median; the cheaper of the two is 3.2 cm).

So the residual is a **concession**, not a reach failure.

## 3. The concession law

```
Δz_clasp ≈ −0.22 · Δh_pair         (vertical yield; taller partner lowers the clasp)
Δλ      ≈  0                      (horizontal yield is zero in practice)
residual after lean ≈ 3.2 cm       (either step-in or clasp lowering, whichever is cheaper)
```

This is a *prior*, not a hard rule: it describes what humans do on average, and the standard
deviation of the buckets is large.

## 4. Three-layer implementation

```
1. GENERATION
   From handshake references, fit the concession law and the joint distribution of
   (clasp height, standoff, lean, extension).
   Output: a prior over negotiation targets, plus a feasibility filter that rejects references
   whose required reach exceeds the robot envelope at any admissible lean.

2. RETARGETING
   Instead of "map the human wrist to the robot wrist", solve for a negotiation target:
     - clasp height = prior mean conditioned on the partner's height
     - standoff     = prior mean conditioned on the height gap (NOT the minimum feasible!)
     - robot lean   = within 22–28°, chosen so that required reach ≤ arm length when possible
   Output: robot trajectory + a CONCESSION BUDGET
     budget = (required reach − arm length) expressed as (step_in_cm, clasp_lower_cm) alternatives

3. DEPLOYMENT
   At runtime, if the partner's actual hand makes the budget non-zero, request the cheaper
   concession (e.g. "lower your hand 3 cm"), monitor compliance, and fall back to re-approach
   if the partner does not comply.
```

The novel interface is the **concession budget**: retargeting stops pretending the robot can
absorb everything, and deployment stops pretending the partner is fixed.

## 5. Experiment plan (kinematic first, physics later)

Three conditions, same 324 handshake references, same robot (G1 kinematics):

| condition | description | expected |
|---|---|---|
| **A. naive retarget** | current-style retargeting, 0° lean, target = human wrist position | low reach rate, large extension overshoot, clasp off the human law |
| **B. contact-preserving retarget** | PAIR-style with a hand-tuned partner weight | contact improves, but the robot sits at its reach limit and extension exceeds 90 % |
| **C. negotiation (ours)** | knobs from the law + concession budget | reach ≈ 100 %, extension within 84–90 %, clasp/standoff/λ within the human-law range |

Metrics (naturalness is judged against the *measured human ranges*, not a distance alone):

| category | metric | human reference |
|---|---|---|
| reach | fraction of contact frames within the arm envelope | — |
| contact | palm-to-palm minimum, contact-frame ratio | 8.4 cm median, 8.4 % of frames |
| symmetry | extension of each side, λ, clasp-height deviation | 84–90 %, 0.49, ~1.00 m |
| naturalness | torso lean inside 22–28°, movement smoothness | 22–28° |
| cost | concession requested (cm), robot displacement (cm) | 3.2 cm order |

## 6. Relation to existing work

- **vs PAIR**: same contact-preserving spirit, but the partner's allowed adaptation is *derived*
  from data instead of hand-tuned, and retargeting is allowed to output infeasibility instead of
  silently stretching the robot.
- **vs AffectiveVLA**: it switches impedance per interaction class; we decide *where* the
  interaction should happen, which is upstream of impedance.
- **vs Handshake + SEA**: mechanical compliance absorbs what geometry cannot. Complementary —
  their SEA handles the residual force, our budget decides whether force should be needed at all.

## 7. Risks

| risk | mitigation |
|---|---|
| adults-only data; robot's ratio at the 4th percentile of observed dyads | state the extrapolation; optionally synthesise dyads with the recovered body-size range (136.7–168.9 cm) |
| no force channel | restrict claims to kinematics; pair with simulation for contact forces |
| "naturalness" has no objective ground truth | use the measured human ranges as acceptance intervals, plus a small user study |
| joint-limit rather than reach may dominate | handle limits explicitly in the solver (see the wrist-target literature) |
| partner may refuse to concede | define a fallback: re-approach, or accept a lower/closer clasp |

## 8. What would falsify the idea

1. If robot reach rate at the human-natural standoff and lean is already ~100 % under naive
   retargeting, there is no asymmetry to correct.
2. If the concession law has huge variance and no predictive value for a robot partner, the
   prior is useless.
3. If human partners refuse or dislike the requested 3 cm, the two-sided protocol fails and only
   one-sided absorption remains — in which case geometry must be replaced by compliance.
