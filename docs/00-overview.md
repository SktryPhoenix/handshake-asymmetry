# Idea in one page

## Problem

A humanoid shaking hands with a real person faces an **inherent asymmetry**:

- different stature (robot 132 cm vs human 150 cm in our reference pair),
- a proportionally shorter arm (74 % of a human's),
- a rigid end effector with no compliance, no touch sensing,
- control latency, and no fingers.

Almost all published methods resolve this by **making the robot absorb the entire mismatch**.
Retargeting keeps the human fixed and bends the robot as far as needed; contact-preserving
retargeting just bends it harder.

This is the wrong object of study. The mismatch is not fatal — **one-sided absorption is**.

## Evidence from real handshakes

Real dyads handle height differences up to 30 cm without difficulty, by splitting the mismatch:

- the clasp sits at an **absolute** height near 100 cm, nearly independent of the height gap;
- horizontally, the clasp sits at the **midpoint** between the two bodies (λ ≈ 0.49);
- both partners **lean forward 22–28°** and **extend their arms to 84–90 %**;
- the taller partner concedes about **6.7 cm** of clasp height across the observed range.

Applying that same split to a humanoid replaces "reach as far as you can" with
"stand like a human would, lean like a human would, and ask the partner for the small residual".

## The three-layer split

| layer | share of the asymmetry |
|---|---|
| generation | learn the concession prior from data (vertical ≈ 0.22 of the gap, horizontal ≈ 0) |
| retargeting | solve a negotiation target (clasp position / standoff / lean) instead of a pure reach target |
| deployment | negotiate the small residual (median 3.2 cm) with the real partner at runtime |

## Numbers that matter

| quantity | value |
|---|---|
| contact frames reachable by a G1 at 0° lean | **32 %** |
| same, at the 25° lean humans actually use | **87 %** |
| residual concession needed after 25° lean | median **3.2 cm** (p90 12.4) |
| released data: P1/P2 forearm ratio at zero betas | **1.000000** (no body-size difference at all) |
| true height range recoverable from `raw_betas` | 136.7 – 168.9 cm |

## Non-goals

- no robot experiments yet; all numbers are kinematic
- no force / tactile modelling (the data has no force channel)
- no claim that 3.2 cm is comfortable — comfort needs a human study
