# PPO course plan

Built with the `course-maker` skill (v2). Separate pages, full-width text|figure split, fiche style.
Shared: `assets/course.css`, `assets/course.js` (nav, predict, timeline engine, chart, bars), `assets/run.js` (TensorBoard export of run `2026-11-25_01-11-15_wheel-limit_vision-memory_center-heading`).
Page sources: `_build/` (lib2.py helpers; q0 = pages 0–1, q1 = pages 2–4, qi = index). Rebuild: `cd _build && python3 q0.py && python3 q1.py && python3 qi.py`.
`draft-v1/` = first short version, kept for reference only.

| # | File | Notion | Status |
|---|---|---|---|
| 0 | 00-mdp.html | RL problem, MDP, Markov, POMDP, terminated vs truncated | done |
| 1 | 01-return.html | return, γ, termination as penalty, reward × dt | done |
| 2 | 02-value.html | V, Q, Bellman, MC vs TD, critic | done |
| 3 | 03-policy.html | Gaussian policy, log-prob, ±1 clamp, actuator lag, entropy | done |
| 4 | 04-gradient.html | log-derivative trick, REINFORCE, noise | done |
| 5 | 05-baseline.html | baseline, advantage, variance proof | next |
| 6 | 06-gae.html | n-step, λ, time-out bootstrap | todo |
| 7 | 07-trust.html | distribution shift, importance ratio, KL | todo |
| 8 | 08-clip.html | clip 4 cases, ε, value clip, entropy term | todo |
| 9 | 09-rslrl.html | epochs, minibatches, adv normalisation, grad norm, adaptive KL bounds | todo |
| 10 | 10-networks.html | sizes, ELU, privileged critic, obs scaling | todo |
| 11 | 11-mini-ppo.html | JS PPO on 1-D lane keeping, live params | todo |
| 12 | 12-read.html | every tag + real run (run.js) | todo |
| 13 | 13-tune.html | method, ablations, symptom table, playground | todo |
| 14 | 14-rewards.html | potential shaping, hacking, boundary case | todo |
| 15 | 15-build.html | design method, randomisation, eval, cheat sheet | todo |

## Verified numbers (node)
- steering: EMA 0.3 + slew 0.1 rad/step: 0→0.45 rad 90 % in 7 steps; reversal 90 % in 12 steps (0.4 s)
- γ 0.99: horizon 100 steps (3.3 s), half-life 69; 0.99^25 = 0.78, 0.9^25 = 0.07
- corridor 11 cells, drift .25/.5/.25, γ .99: V centre 29.83, next to line 9.64; γ .9: 8.28
- REINFORCE toy (μ .6, a* .1, σ .2): batch 1 sd 2.37, 51 % wrong sign; batch 8 sd .84, 6.6 %; with baseline batch 1 sd 1.75, 7.2 %
- clamp: μ .9 σ .2 → 30.9 % clipped; σ 1 μ 0 → 31.7 %
- current reward best per step 7.5 × 1/30 = 0.25 → return bound 25 (logged run: 13.3)

## Config at time of writing (re-read before continuing)
centering 1.0 (std .25), progress 2.0, heading .5, slow_speed −1 (1.5 m/s), boundary −2 (margin .50), steering_smoothness −.15, steering_rate −.1; max_speed 3.0; lap_completed termination present but never fires (detector lived in removed reward).
