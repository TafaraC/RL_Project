# Game split

The split was recorded on 3 October 2026, before any baseline results existed.

## Method

The 25 public ARC-AGI-3 games, in alphabetical order:

```
ar25 bp35 cd82 cn04 dc22 ft09 g50t ka59 lf52 lp85 ls20 m0r0 r11l
re86 s5i5 sb26 sc25 sk48 sp80 su15 tn36 tr87 tu93 vc33 wa30
```

These were shuffled with a fixed seed. The first four became the development set and the next three the held-out set. No game was chosen by hand or by difficulty.

```python
import random
games = [...]  # the 25 ids above, in the order shown
random.Random(2026).shuffle(games)
dev, held_out = sorted(games[:4]), sorted(games[4:7])
```

## Sets

| Set | Games | Use |
|---|---|---|
| Development | dc22, ft09, g50t, m0r0 | Training, debugging, choosing improvements and tuning |
| Held-out | cd82, tr87, vc33 | Final transfer test only. Never trained or tuned on |
| Unused | the other 18 games | Not used |

## Rules

- Held-out games are evaluated once per final agent version, after design decisions are frozen.
- Baselines and improved agents are compared on the same games, with the same action budget and seeds.

## Held-out evaluation

Fixed on 27 October 2026, before any held-out result was observed:
- All three ft09-trained seeds of the baseline, novelty and final memory stages (DQN no-change memory; PPO memory + novelty).
- Five frozen rollouts per checkpoint per game, 1,000 actions each, with sampling seeds 100–104.
- PPO samples from its policy; DQN uses epsilon 0.3.
- No parameter updates, no held-out source inspection, and no model selection from these results.
