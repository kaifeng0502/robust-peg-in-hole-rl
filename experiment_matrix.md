# Evaluation matrix

All rows use the same 500 held-out initial states per evaluation seed. Keep the Factory success condition fixed: peg XY error below 2.5 mm and peg base within 1 mm of the socket bottom.

| Method | Pose randomization in training | Friction randomization in training | Evaluation distribution |
|---|---:|---:|---|
| Impedance + fixed spiral | n/a | n/a | pose + friction |
| PPO nominal | off | off | pose + friction |
| PPO pose only | on | off | pose + friction |
| PPO friction only | off | on | pose + friction |
| PPO full | on | on | pose + friction |

Report these task metrics for every method:

- success rate with a Wilson 95% interval;
- mean and median completion time among successful trials;
- maximum insertion depth reached;
- peak commanded force (clearly labeled as a command, until a real contact sensor is added).

Report these learning metrics for every PPO row:

- environment steps to 50%, 70%, and 80% held-out success;
- evaluation success versus environment steps;
- mean and standard deviation across seeds 42, 43, and 44;
- training reward and critic loss curves for diagnosing instability.

The main comparison is fixed spiral versus PPO full. The other PPO rows form the domain-randomization ablation and should not be presented as separate headline methods.
