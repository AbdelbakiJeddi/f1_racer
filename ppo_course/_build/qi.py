from lib2 import *
parts=[("00-mdp","0 · The RL problem","MDP, Markov property, partial observation, terminated vs truncated",1),
("01-return","1 · Reward and return","discount γ derived three ways, termination as hidden penalty",1),
("02-value","2 · Value, Bellman, TD","V and Q, Bellman equation, Monte Carlo vs TD, the critic",1),
("03-policy","3 · The stochastic policy","Gaussian, log-prob, ±1 clamp, actuator lag, entropy, σ",1),
("04-gradient","4 · The policy gradient","log-derivative trick, REINFORCE, why it is noisy",1),
("","5 · Baseline and advantage","why subtracting V is unbiased and cuts variance",0),("","6 · GAE","n-step returns, λ, bootstrapping on time-outs",0),
("","7 · Trust region","distribution shift, importance ratio, KL",0),("","8 · The PPO clip","the four cases, why ε = 0.2, value clip",0),
("","9 · PPO in rsl_rl","epochs, mini-batches, grad-norm, adaptive KL",0),("","10 · Networks","sizes, ELU, input scaling",0),
("","11 · Live mini-PPO","a car trained with PPO inside the page",0),("","12 · Reading data","every TensorBoard tag, your real run",0),
("","13 · Tuning","method, ablations, symptom table",0),("","14 · Reward design","shaping, hacking, the boundary case",0),("","15 · Build + synthesis","design method, cheat sheet",0)]
cards='<div class="grid">'+"".join((f'<a class="card" href="{f}.html" style="text-decoration:none;color:inherit"><h4>{t}</h4><span class="small">{d}</span></a>' if ok else f'<div class="card" style="opacity:.55;border-style:dashed"><h4>{t}</h4><span class="small">{d} · coming next</span></div>') for f,t,d,ok in parts)+'</div>'
b=split("""<p class="lead">A course on PPO built on one real robot: your F1 racer in Isaac Lab. Every idea is derived from first principles, broken in an interactive figure, then found in the project's code.</p>
<p>How each notion is built: <b>problem</b> → <b>derivation</b> → <b>figure</b> → <b>break it</b> (predict first) → <b>why this value</b> → <b>to remember</b> → <b>in the project</b> → deep dive.</p>""",
key("PPO tries slightly different actions, measures which ones paid more than expected, and makes those a bit more likely, never moving the policy too far in one step. Pages 0–4 build everything that sentence needs."))+cards
page("index","PPO, learned on an F1 racer","From the RL problem to tuning and designing your own policy. Numbers read from the project's current config and logs.",b,"")
