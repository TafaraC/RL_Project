# Game split (ARC-AGI-3 public set)

Recorded 3 October 2026, before any baseline results were produced.

## Method

The public set has 25 games (ids fetched from the ARC-AGI-3 API via
`arc_agi.Arcade().get_environments()`):

```
ar25 bp35 cd82 cn04 dc22 ft09 g50t ka59 lf52 lp85 ls20 m0r0 r11l
re86 s5i5 sb26 sc25 sk48 sp80 su15 tn36 tr87 tu93 vc33 wa30
```

The list above (alphabetical order) was shuffled with
`random.Random(2026).shuffle(...)`. The first 4 games became the
development set and the next 3 the held-out set. No game was chosen by hand
or by looking at its difficulty.

```python
import random
games = [...]  # the 25 ids above, in the order shown
random.Random(2026).shuffle(games)
dev, held_out = sorted(games[:4]), sorted(games[4:7])
```

## Sets

| Set | Games | Use |
|---|---|---|
| Development | `dc22`, `ft09`, `g50t`, `m0r0` | Training, debugging, choosing improvements, tuning |
| Held-out | `cd82`, `tr87`, `vc33` | Final generalisation check only. Never trained on, never tuned on |
| Unused | the other 18 public games | Not used so far |

## Rules

- Held-out games are evaluated once per final agent version, after design
  decisions are frozen. Do not use them to pick between improvements.
- Any change to these sets, or any use of an unused game, must be added to
  the log below with the date and the reason. The report must disclose every
  game used during development.
- Baselines and improved agents are compared on the same games, with the same
  action budget and seeds.

## Change log

| Date | Change | Reason |
|---|---|---|
| 2026-10-03 | Initial split | See method above |
