# Methods survey: how human–robot interaction papers handle the mismatch

34 works collected (24 focused on human–robot / human–humanoid interaction; the rest are the
retargeting / generation / prediction foundations they build on). Full per-paper index, arXiv
IDs and notes live in the companion private library; this document is the analytical summary.

## 1. Five method families

| family | core idea | representatives |
|---|---|---|
| **contact-centric retargeting** | treat the contact *relationship* (distance matrices, semantic contact, relative configuration) as the explicit objective rather than joint angles | PAIR, VICAR, HumiMimic, REACT, CoCarry |
| **reactive closed loop** | perceive and react to the partner online: intent inference, partner prediction, impedance switching | RHINO, Social Humanoid, CoDance, AffectiveVLA |
| **morphology transfer** | translate human data into a robot-specific reference (or search for the reaction that fits the embodiment) | HumiMimic, EgoAlign, Embodiment-Aware Search, Tokens of Motion, Hand Reaching |
| **physics / force / safety** | absorb interaction forces mechanically or with force-aware data | ThorArena, Learning to Assist, Handshake + SEA |
| **high-level coordination** | planning, role assignment, social navigation | RoboAssist, Networked Humanoid, SocialCoAct, Can I Join You, Pairwise HHI, Trust-Driven Role Adaptation |

## 2. The central table: who adapts?

| adaptation is allocated to… | papers | notes |
|---|---:|---|
| **robot absorbs everything, partner fixed** | **14** | the default assumption of the field |
| partner is predicted / modelled, but not asked to move | 4 | REACT's generative partner model, Social Humanoid's occlusion prediction, CoDance, AffectiveVLA |
| partner may adapt — but the weight is **hand-tuned** | **1** | PAIR's `L_hum` / `w_hum` |
| explicit role assignment (task level, not contact level) | 2 | Trust-Driven Role Adaptation; Handshake + SEA (mechanical compliance) |
| both agents symmetric | 1 | CoCarry (but it is robot–robot) |

Take-away: **nobody learns how much the partner should yield.** The only mechanism that lets the
partner move carries a manually chosen weight.

## 3. The one that comes closest

**PAIR / Beyond Mimicry** (Huang et al.): human–human demos → contact-preserving two-stage
retargeting (`L_retarget = w_con L_con + w_kin L_kin + w_hum L_hum + w_reg L_reg`) → a hierarchical
policy (D-STAR: phase attention for *when*, multi-scale spatial module for *where*, diffusion head).
It explicitly shows that standard retargeting breaks the contacts, and it is the only work in the
survey that spans generation, retargeting and deployment.

Its weakness is precisely our target: `w_hum` is a constant.

Other close neighbours:
- **Hand Reaching for Human-Robot Handshaking** — the only paper dedicated to handshake *approach*;
  learns reaching from third-person human–human data for non-backdrivable robots. It stops before
  contact.
- **AffectiveVLA** — treats handshake / fist bump / hug as explicit tasks with intent gating and
  per-interaction impedance profiles (hand-set profiles).
- **Handshake + SEA** — the only physical handshake experiment: series-elastic elements absorb
  impact, measured through safety/comfort perception.

## 4. Layer attribution (generation / retargeting / deployment)

| layer | papers touching it | papers *only* in that layer | representatives |
|---|---:|---:|---|
| generation | 9 | 4 | EgoAlign, ThorArena, SocialCoAct, HiPHI |
| retargeting | 10 | **1** | Hand Reaching |
| deployment | 15 | 9 | RHINO, AffectiveVLA, CRISP, SEA, Trust-Driven |
| **all three** | **1** | — | **PAIR** |

Two observations:

1. Retargeting is crowded but has essentially stopped existing as a standalone contribution — it
   is now a component inside generation or deployment pipelines.
2. Deployment is the largest group, yet 9 of those papers never touch morphology mismatch (they are
   social / planning / control work).

## 5. The gap we target

1. **Generation does not know whether the downstream robot can execute the reference.** PAIR
   generates first, retargets later, and nobody is responsible when the retargeting turns out
   infeasible. A reachability-aware generator/filter is missing.
2. **Retargeting never emits a "concession budget".** It absorbs everything and reports a
   trajectory; it does not say "I can take 3.6 cm, the rest must come from the partner".
3. **Deployment has no interface for a concession request.** Existing deployment layers accept
   trajectories or intents, not "please lower your hand by 3 cm".
