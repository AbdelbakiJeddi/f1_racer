from lib2 import *

# ================= PAGE 0 : MDP =================
b = sec("Driving without a driving rule","the problem", split(
"""<p class="lead">You want the car to stay between two yellow lines. The classic way is to write the rule yourself: a PID on the lateral error, gains tuned by hand.</p>
<p>Machine learning offers a second way: <b>supervised learning</b>. Show the network thousands of camera frames, each labelled with <i>the correct steering</i>. But nobody has those labels. A human driver would have to drive every situation, including the near-crashes you most need.</p>
<p>What you <i>can</i> say is how good an outcome was: "you stayed centred", "a wheel touched the line". <b>Reinforcement learning (RL)</b> learns from exactly that: a score, not a correct answer.</p>""",
fig("cmp","Supervised learning needs the answer. RL only needs a score, given after the action."))+
key("RL replaces <i>labels</i> (what to do) with <i>rewards</i> (how good it was). Everything else in this course is about turning that weak signal into a good policy."))

b += sec("The loop, word by word","agent ↔ environment", split(
"""<p>RL is one loop, repeated. Each word below is used for the rest of the course.</p>
"""+table(["Word","Meaning","Your car"],[
["<b>agent</b>","the decision maker","the actor network"],
["<b>environment</b>","everything else: physics, track, camera","Isaac Sim, 64 copies in parallel"],
["<b>state</b> s","the full situation of the world","exact pose, speeds, actuator states"],
["<b>observation</b> o","what the agent is allowed to see","7 numbers from camera + sensors"],
["<b>action</b> a","what the agent outputs","steer, throttle in −1…1"],
["<b>reward</b> r","one number: how good this step was","sum of the reward terms"],
["<b>step</b>","one turn of the loop","1/30 s"],
["<b>episode</b>","steps from a reset to an end","≤ 40 s = 1200 steps"],
["<b>policy</b> π","the rule obs → action that we learn","the actor network's output"]]),
fig("loop","One step. The environment advances the simulation by 1/30 s, then answers with a reward and a new observation.")))

b += sec("The MDP: five ingredients","formal model", split(
"""<p>The loop is formalised as a <b>Markov Decision Process (MDP)</b>. Every RL algorithm, PPO included, assumes these five pieces exist, even if it never writes them down.</p>"""+
table(["Symbol","Name","In the F1 racer","Who knows it?"],[
["S","states","car pose, velocities, steering and throttle actuator states, track geometry","the simulator"],
["A","actions","2 numbers, clamped to −1…1","the agent"],
["P(s′|s,a)","dynamics","PhysX at 120 Hz, 4 sub-steps per action","nobody (black box)"],
["R(s,a)","reward function","6 weighted terms × dt","you wrote it"],
["γ","discount","0.99","you chose it"]])+
"""<p>The key line is <b>P</b>: PPO never sees the physics equations. It only sees samples (s, a, r, s′). That is why it is called <b>model-free</b>.</p>""",
fig("mdp","An MDP as a graph: from a state, an action leads to a random next state and a reward.")+
key("PPO is model-free: it learns from samples of the dynamics, never from the equations. Your simulator could be swapped for a real car without changing the algorithm.")))

b += sec("The Markov property","why the observation list matters", split(
"""<p>An MDP assumes the <b>Markov property</b>:</p>"""+eq("P(s<sub>t+1</sub> | s<sub>t</sub>, a<sub>t</sub>) = P(s<sub>t+1</sub> | s<sub>t</sub>, a<sub>t</sub>, s<sub>t−1</sub>, a<sub>t−1</sub>, …)")+
"""<p>In words: <b>the present contains everything needed to predict the future</b>. History adds nothing.</p>
<p>Why the agent cares: the policy sees one observation and must choose one action. If two situations look identical but need different actions, <i>no</i> policy can handle both. Learning cannot fix missing information.</p>
<h3>Break it</h3>
<p>Two cars start on the centre line (e_lat = 0). Car A points 0.25 rad left, car B 0.25 rad right. Same speed, 2.5 m/s. A simple controller drives both. Choose what it can see.</p>"""+pred("pr0"),
widget("<label>observation <select id='obs'><option value='lat'>e_lat only</option><option value='both'>e_lat + e_heading</option></select></label>",
"<div id='mk-tl'></div><div id='mk'></div><div class='read' id='mk-t'></div>")+
"<p class='small'>Kinematic bicycle model, wheelbase 0.32 m, steer = −1·e_lat (−1·e_heading). Dashed orange: car centre at 0.42 m = a wheel touches the line (0.56 m).</p>")+
why("e_heading and the speeds are in your observation","A camera frame gives a position. Position alone is not Markov for a moving car: you also need where it is going. e_heading, vx, vy and wz are the 'derivative' information that makes the observation (nearly) Markov.")+
key("If the observation is not Markov, the policy must guess. Check your observation list first, before any tuning."))

b += sec("State vs observation: your car is partially observed","POMDP", split(
"""<p>The actor does not get the state. It gets a camera-derived <b>observation</b>. When observation ≠ state, the problem is a <b>POMDP</b> (Partially Observable MDP).</p>
<p>What is hidden from the actor in your project:</p>
<ul><li>the exact track geometry ahead (only two scan bands of the image)</li>
<li>the <b>throttle actuator state</b> (rate-limited internally, only its effect vx is seen)</li>
<li>the true lateral error when no line is visible (the detector then answers from memory)</li></ul>
<p>Your detector's <b>memory buffer</b> is a small history inside the observation function. It turns 'no line visible' into 'last known centre', which restores part of the Markov property.</p>""",
fig("pomdp","Asymmetric actor–critic: the critic is trained on the true state, the actor drives from what it can see.")+
table(["Actor sees (7)","Critic sees (8)"],[["e_lat (camera)","lateral error (exact)"],["e_heading (camera)","heading error (exact)"],["confidence","vx, vy, wz"],["vx, vy, wz","steering angle"],["steering angle","speed along the track"],["",'normalised lateral error']]))+
why("the critic may see more than the actor","The critic is only used during training to judge actions. A better judge gives better learning signals. At deployment only the actor runs, so its inputs must exist on the real car. The critic's do not.")+
anchor([(P+"mdp/observations.py:94","vision_track_state: the actor's 7 numbers"),(P+"mdp/observations.py:54","privileged_track_state: the critic's 8 numbers"),(P+"mdp/observations.py:119","detector memory reset at each new episode")]))

b += sec("Time: why 30 decisions per second","dt, decimation", split(
"""<p>Physics runs at <b>120 Hz</b> (dt = 1/120 s). The policy decides at <b>30 Hz</b>: each action is held for 4 physics steps (<b>decimation = 4</b>).</p>
<p>Two different needs:</p>
<ul><li><b>Physics</b> needs small steps to stay stable (wheel contacts, suspension).</li>
<li><b>Decisions</b> should be as slow as the task allows. Every extra decision per second makes the episode longer in steps, so rewards that matter are more steps away.</li></ul>
<p>The camera renders once per decision, which is also the most expensive part of a training iteration.</p>""",
table(["Decision rate","Steps in 40 s","γ for a 3.3 s horizon"],[["120 Hz","4800","0.9975"],["<b>30 Hz (yours)</b>","<b>1200</b>","<b>0.99</b>"],["10 Hz","400","0.97"]])+
"<p class='small'>Horizon in steps = 1/(1−γ), explained on page 1. Same look-ahead in seconds needs a γ that depends on the decision rate.</p>")+
why("decimation 4","The steering servo is already rate-limited to 3 rad/s (0.1 rad per decision). Deciding faster than the actuator can follow adds steps without adding control.")+
anchor([(P+"config/f1_car/env_cfg.py:265","sim dt = 1/120"),(P+"config/f1_car/env_cfg.py:283","decimation = 4 → 30 Hz control"),(P+"config/f1_car/env_cfg.py:284","episode_length_s = 40")]))

b += sec("How an episode ends: terminated vs truncated","the most misunderstood detail", split(
"""<p>An episode ends for one of two very different reasons:</p>"""+
table(["","Terminated","Truncated"],[["meaning","the MDP really ended","we stopped watching"],["your terms","off_track, rollover, lap_completed","time_out (40 s)"],["future reward","0, by definition","unknown but not 0: the car could have driven on"],["what the algorithm does","no future value","estimate the future with the critic: <b>bootstrap</b>"]])+
"""<p>Mixing them up is a classic bug: if a time-out counted as a crash, a car that drives perfectly for 40 s would be punished at the end for something that is not a failure.</p>
<p>Isaac Lab passes the time-out flag to rsl_rl, which adds γ·V(s) to the last reward of truncated episodes. Page 2 shows why that is exactly the missing future.</p>""",
fig("ends","Same final position, two endings, two different learning signals.")+
"""<div class='why'><b>Watch out: lap_completed.</b> It is a termination, so finishing a lap <i>ends the reward stream</i>. With a positive reward every step, completing a lap means losing all future reward: the agent has no reason to want it. Also, lap detection lived inside the removed lap_completion reward, so in your current config this term never fires.</div>""")+
anchor([(P+"config/f1_car/env_cfg.py:242","time_out = DoneTerm(..., time_out=True): marks truncation"),("isaaclab_rl/rsl_rl/vecenv_wrapper.py:166",'extras["time_outs"] = truncated'),("rsl_rl/algorithms/ppo.py:154","on time-outs: reward += γ·V(s)"),(P+"mdp/terminations.py:41","off_track: any wheel corner ≥ 0.56 m")]))

b += sec("Why 64 cars at once","parallel environments", split(
"""<p>All 64 cars are independent copies of the same MDP. They share one policy.</p>
<ul><li><b>Throughput.</b> One GPU renders 64 cameras and simulates 64 cars per step: about 1050 samples/s in your run.</li>
<li><b>Decorrelation.</b> Consecutive steps of one car are almost identical. 64 cars give 64 different situations per step, which makes each update less biased toward one moment.</li></ul>""",
"<p>Your config file defaults to 32 (<code>F1RacerSceneCfg(num_envs=32)</code>); the training command overrides it with <code>--num_envs 64</code>.</p>")+
deep("the formal definitions", """<p><b>MDP</b> = (S, A, P, R, γ). A <b>policy</b> π(a|s) is a probability of each action in each state. The goal is the policy that maximises the expected discounted return J(π) = E[Σ γ<sup>t</sup> r<sub>t</sub>] (page 1).</p>
<p><b>POMDP</b> adds an observation function O(o|s). Standard fixes when the observation is not Markov: stack the last k observations, give the policy a memory (recurrent network, LSTM/GRU), or engineer features that carry the missing information. Your project uses the third option (e_heading, speeds) plus the detector memory.</p>
<p>In Isaac Lab, <code>terminated</code> and <code>truncated</code> follow the Gymnasium API: <code>step()</code> returns both flags separately so algorithms can bootstrap only on truncation.</p>"""))

s0 = r'''
svg('cmp',640,210, box(10,20,180,60,['Supervised','frame → correct steer'])+L([[190,50],[246,50]],{arrow:true})+box(250,20,160,60,['network'])+L([[410,50],[466,50]],{arrow:true})+box(470,20,160,60,['compare to label'],{col:C.mut})+
 T(320,100,'needs a teacher for every frame',{col:C.acc})+
 box(10,120,180,60,['RL','frame → try a steer'],{col:C.acc})+L([[190,150],[246,150]],{arrow:true})+box(250,120,160,60,['environment'])+L([[410,150],[466,150]],{arrow:true,col:C.good})+box(470,120,160,60,['score: reward'],{col:C.good})+T(320,200,'needs only a score, after the fact',{col:C.good}));
svg('loop',640,230, box(20,80,160,70,['Agent','actor network'],{col:C.acc})+box(460,80,160,70,['Environment','Isaac Sim'])+
 L([[100,80],[100,34],[540,34],[540,76]],{arrow:true,col:C.acc})+T(320,26,'action a_t: steer, throttle',{col:C.acc,b:true})+
 L([[540,150],[540,200],[100,200],[100,154]],{arrow:true,col:C.blue})+T(320,193,'observation o_t+1: 7 numbers',{col:C.blue,b:true})+
 L([[460,115],[186,115]],{arrow:true,col:C.good})+T(320,107,'reward r_t: one number',{col:C.good,b:true})+T(320,135,'1/30 s per turn',{col:C.mut,size:11}));
svg('mdp',640,230, box(20,90,110,50,['state s'])+L([[130,115],[226,115]],{arrow:true,col:C.acc})+T(178,107,'action a',{col:C.acc})+`<circle cx="240" cy="115" r="10" fill="${C.acc}"/>`+
 L([[250,110],[400,45]],{arrow:true})+L([[250,115],[400,115]],{arrow:true})+L([[250,120],[400,185]],{arrow:true})+
 box(405,20,120,50,["s′₁","p = 0.6"])+box(405,90,120,50,["s′₂","p = 0.3"])+box(405,160,120,50,["s′₃","p = 0.1"])+
 T(580,49,'r = 0.21',{col:C.good})+T(580,119,'r = 0.18',{col:C.good})+T(580,189,'r = 0.02',{col:C.good})+T(320,225,'illustrative probabilities: the agent never sees them, only samples',{col:C.mut,size:11}));
predict('pr0','both cars see e_lat = 0 at the start. With e_lat only, can one policy keep both on the track?',
 ['Yes, with good enough gains','No: same observation, same action, but they need opposite steering','Only if the speed is lower'],1,
 'At t = 0 the two cars are indistinguishable, so the policy must output the same steer for both, though A needs to steer right and B left. Larger gains only make the oscillation faster: with e_lat only, steer gain 1 touches the line after 116 steps (3.9 s), gain 2 after 72. Adding e_heading keeps both within 6 cm.');
function sim(psi0, useH){ const dt=1/30, Lb=0.32, v=2.5; let y=0, psi=psi0; const p=[[0,0]]; for(let k=0;k<180;k++){ let d=-y-(useH?psi:0); d=Math.max(-.45,Math.min(.45,d)); y+=v*Math.sin(psi)*dt; psi+=v/Lb*Math.tan(d)*dt; p.push([(k+1)/30,y]); } return p; }
let A,B; const mk=timeline('mk-tl',()=>181,k=>{
 const cut=p=>p.slice(0,k+1);
 chart('mk',[{pts:cut(A),col:C.acc,label:'car A (points left)'},{pts:cut(B),col:C.blue,label:'car B (points right)'},{pts:[[0,.42],[6,.42]],col:C.acc,dash:true},{pts:[[0,-.42],[6,-.42]],col:C.acc,dash:true}],{xmin:0,xmax:6,ymin:-1.2,ymax:1.2,xl:'time (s)',yl:'lateral position of the car centre (m)'});
 const t=(p)=>{const i=p.findIndex(q=>Math.abs(q[1])>=.42); return i<0?'never':`${(i/30).toFixed(1)} s`;};
 $('mk-t').textContent=`car A touches at: ${t(A)}   car B touches at: ${t(B)}`;
 return `t = ${(k/30).toFixed(2)} s`; },{ms:30});
bind(['obs'],v=>{ A=sim(0.25,v.obs==='both'); B=sim(-0.25,v.obs==='both'); mk.redraw(); });
svg('pomdp',640,240, box(10,20,140,55,['Camera','128×128 RGB'])+L([[150,47],[186,47]],{arrow:true})+box(190,20,150,55,['Detector + memory','e_lat, e_heading, conf'])+L([[340,47],[386,47]],{arrow:true})+box(390,20,120,55,['Actor','sees 7'],{col:C.acc})+L([[510,47],[546,47]],{arrow:true})+box(550,20,80,55,['action'])+
 box(190,150,150,55,['Simulator','exact state'])+L([[340,177],[386,177]],{arrow:true})+box(390,150,120,55,['Critic','sees 8'],{col:C.blue})+L([[510,177],[546,177]],{arrow:true})+box(550,150,80,55,['value V'])+
 T(450,105,'deployed on the car',{col:C.acc,size:11})+T(450,232,'training only',{col:C.blue,size:11}));
svg('ends',640,200, L([[20,60],[600,60]],{col:C.line,sw:6})+box(10,40,70,40,['reset'])+box(500,10,130,46,['off_track','terminated'],{col:C.acc})+T(565,72,'future = 0',{col:C.acc,b:true})+
 L([[20,150],[600,150]],{col:C.line,sw:6})+box(10,130,70,40,['reset'])+box(500,100,130,46,['time_out','truncated'],{col:C.good})+T(565,162,'future ≈ V(s)',{col:C.good,b:true})+
 L([[600,150],[635,150]],{col:C.good,dash:true,arrow:true})+T(300,190,'the car on the second line could have kept driving',{col:C.mut,size:11}));
'''
page("00-mdp","0 · The RL problem","From 'no labels' to a Markov Decision Process, with your car as the example.",b,s0)

# ================= PAGE 1 : RETURN =================
b = sec("One reward is not enough","the problem", split(
"""<p class="lead">The car steers now; a wheel touches the line 25 steps later. The reward of <i>this</i> step says nothing about that.</p>
<p>We need one number that scores an action by <b>everything that follows it</b>. That number is the <b>return</b>.</p>""",
read("heading error 0.2 rad at 2.5 m/s → drifts sideways at 0.50 m/s\\ncentre line → wheel on the line (car centre at 0.42 m): 0.85 s = 25 steps")))

b += sec("Building the return","from first principles", split(
"""<h3>Attempt 1: add all future rewards</h3>"""+eq("G<sub>t</sub> = r<sub>t</sub> + r<sub>t+1</sub> + r<sub>t+2</sub> + …")+
"""<p>Two problems. For a car that never crashes, the sum grows without limit, so two good policies cannot be compared. And a reward 1000 steps away counts as much as the next one, though it is far less certain and far less related to the current action.</p>
<h3>Attempt 2: discount the future</h3>"""+eq("G<sub>t</sub> = r<sub>t</sub> + γ·r<sub>t+1</sub> + γ²·r<sub>t+2</sub> + …  ,   0 ≤ γ < 1")+
"""<p>Each step further away is multiplied once more by γ. With γ = 0.99, a reward 69 steps ahead counts half.</p>
<h3>The recursive form (used everywhere later)</h3>"""+eq("G<sub>t</sub> = r<sub>t</sub> + γ·(r<sub>t+1</sub> + γ·r<sub>t+2</sub> + …) = r<sub>t</sub> + γ·G<sub>t+1</sub>")+
"<p>'Return now = reward now + γ × return from the next step.' Page 2 turns this line into the Bellman equation.</p>",
fig("disc","Weight of each future reward, γᵗ, for three values of γ.")))

b += sec("Three ways to read γ","why discount at all", split(
table(["Reading","Statement","For γ = 0.99"],[
["<b>Math</b>","the sum is bounded: G ≤ r<sub>max</sub>/(1−γ)","at most 100 × the best reward"],
["<b>Probability</b>","as if the world could end with probability 1−γ at each step","a 1 % 'sudden end' risk per step"],
["<b>Horizon</b>","roughly how far ahead the agent looks: 1/(1−γ) steps","100 steps = 3.3 s; half-life 69 steps = 2.3 s"]]),
table(["γ","Horizon (steps)","Horizon (s at 30 Hz)","Half-life (steps)"],[["0.9","10","0.33","6.6"],["0.95","20","0.67","13.5"],["<b>0.99</b>","<b>100</b>","<b>3.33</b>","<b>69</b>"],["0.995","200","6.67","138"],["0.999","1000","33.3","693"]]))+
why("γ = 0.99 for this car","The danger you care about (heading error growing into a line touch) unfolds over about 25 steps. Weight of a reward 25 steps ahead: γ = 0.9 → 0.07 (almost invisible), γ = 0.95 → 0.28, γ = 0.99 → 0.78 (clearly felt). Going higher (0.999) makes the critic predict 33 s ahead, where the future is mostly noise: learning gets slower and noisier for no driving benefit."))

b += sec("Break it: what does falling cost?","interactive",
split(pred("pr1")+widget(rng("g","γ ",0.8,0.999,0.001,0.99)+rng("fall","car fails at step ",5,300,1,90)+rng("rps","reward per step ",0.05,0.3,0.01,0.25),"<div id='ret'></div><div class='read' id='ret-t'></div>"),
"""<p>Orange: how much each future step counts (γ<sup>t</sup>). Green: what the car really collects before it fails.</p>
<p>Default reward per step = 0.25: the best your <b>current</b> reward can give per step (see below).</p>
<p>Notice that a failure is never punished <i>explicitly</i> in your config. The cost is the green area that is cut off: all the future that will never be collected.</p>""")+
key("Termination is a penalty in disguise: it costs the whole discounted future. That is why a car with only positive rewards still learns to avoid crashing."))

b += sec("Your reward, as the return sees it","current config", split(
"""<p>The reward manager computes each term, multiplies by its weight, sums, then multiplies by <b>dt = 1/30</b>. So per-step rewards are small, and returns stay in a readable range.</p>"""+
table(["Term","Weight now","Best per step (before dt)"],[["centering","1.0","1.0 (exactly on the centre line)"],["progress","2.0","6.0 (3 m/s along the track)"],["heading","0.5","0.5 (aligned)"],["slow_speed","−1.0","0 above 1.5 m/s"],["boundary","−2.0","0 while the outer wheel is within 0.50 m"],["steering_smoothness, steering_rate","−0.15, −0.1","0 when steering is still"]])+
read("best per step  = (1.0 + 6.0 + 0.5) × 1/30 = 0.25\\nbest return    = 0.25 / (1 − 0.99)        = 25\\nlogged run (centering 3.5, no progress): 4 × 1/30 = 0.133 → 13.3"),
"""<p><b>Speed now dominates.</b> At 3 m/s, progress gives 6 of the 7.5 points per step. Centering gives at most 1. The return will prefer a fast car that is roughly centred over a slow, perfectly centred one.</p>
<p>The trained run you analysed used a different set (centering 3.5, no progress): returns of ≈ 13, and no reason to drive faster than 1.5 m/s.</p>""")+
why("multiply by dt","It makes the return independent of the control rate. Doubling the decision rate doubles the number of steps but halves dt, so seconds of good driving keep the same value. It also keeps values small (25, not 750), which is easier for the critic network.")+
anchor([(P+"config/f1_car/env_cfg.py:204","centering, weight 1.0, std 0.25"),(P+"config/f1_car/env_cfg.py:208","progress, weight 2.0"),("isaaclab/managers/reward_manager.py:151","sum of weighted terms × dt"),(P+"config/f1_car/agents/rsl_rl_ppo_cfg.py:38","gamma = 0.99")]))

b += sec("Positive or negative rewards: it changes the behaviour","sign matters", split(
'<div class="grid2"><div class="card"><h4><span class="ok">Mostly positive per step</span></h4>Every extra step alive pays. Ending the episode loses future reward. <b>Behaviour:</b> avoid ending. <b>Risk:</b> stall or crawl if survival alone pays enough.</div><div class="card"><h4><span class="ko">Mostly negative per step</span></h4>Every step costs. Ending early stops the costs. <b>Behaviour:</b> the agent learns to crash fast. This is why "−1 per step" designs need a big explicit failure penalty.</div></div>',
"<p>Your config is mostly positive (centering, progress, heading). Failing costs the future (up to 25), so the termination itself does the job of a crash penalty.</p>"))

b += deep("the objective PPO maximises", """<p>The quantity PPO tries to increase is the <b>expected discounted return</b> from the start:</p>"""+eq("J(π) = E<sub>τ∼π</sub> [ Σ<sub>t</sub> γ<sup>t</sup> r<sub>t</sub> ]")+
"""<p>The expectation is over trajectories τ = (s₀, a₀, r₀, s₁, …) produced by running π in the environment: random because the policy samples actions and the world may be random. Two policies are compared by this single number.</p>
<p>Why use γ < 1 even though episodes end after 1200 steps? Without it, the critic would have to predict up to 1200 steps of reward, and the time left in the episode would change values of otherwise identical states. The discount keeps the problem 'local' and time-independent.</p>
<p>Alternatives exist (average-reward RL, finite-horizon returns), but discounted returns are what PPO, rsl_rl and nearly all robotics RL use.</p>""")

s1 = r'''
const pts=g=>{const p=[];for(let t=0;t<=300;t++)p.push([t,Math.pow(g,t)]);return p;};
chart('disc',[{pts:pts(0.9),col:C.mut,label:'γ = 0.9'},{pts:pts(0.95),col:C.blue,label:'γ = 0.95'},{pts:pts(0.99),col:C.acc,label:'γ = 0.99'}],{ymin:0,ymax:1,xl:'steps ahead',vlines:[{x:25,label:'25 steps: line touch'}]});
predict('pr1','with γ = 0.99 and 0.25 per step, the car fails at step 90. Roughly what share of its possible return does it lose?',['about 10 %','about 40 %','about 90 %'],1,
 'Lost = γ⁹⁰ of the total = 0.99⁹⁰ ≈ 0.40. The 90 steps it did drive are worth 60 % of the infinite future: 14.9 of 25.');
bind(['g','fall','rps'],v=>{
 const w=[],a=[]; for(let t=0;t<=300;t++){ w.push([t,Math.pow(v.g,t)]); a.push([t,t<v.fall?Math.pow(v.g,t):0]); }
 chart('ret',[{pts:w,col:C.acc,label:'weight γᵗ'},{pts:a,col:C.good,label:'actually collected'}],{ymin:0,ymax:1,xl:'steps ahead'});
 const full=v.rps/(1-v.g), got=v.rps*(1-Math.pow(v.g,v.fall))/(1-v.g);
 $('ret-t').textContent=`horizon ${(1/(1-v.g)).toFixed(0)} steps = ${(1/(1-v.g)/30).toFixed(1)} s     half-life ${(Math.log(.5)/Math.log(v.g)).toFixed(0)} steps\nreturn if it never fails : ${full.toFixed(2)}\nreturn if it fails at ${v.fall}: ${got.toFixed(2)}     lost: ${(full-got).toFixed(2)} (${(100*(full-got)/full).toFixed(0)} %)`;
});
'''
page("01-return","1 · Reward and return","The car does not maximise the next reward. It maximises the discounted sum of everything after.",b,s1)
