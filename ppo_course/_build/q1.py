from lib2 import *
PPO="rsl_rl/algorithms/ppo.py"
# ================= PAGE 2 : VALUE =================
b = sec("Waiting for the end is too slow","the problem", split(
"""<p class="lead">The return G<sub>t</sub> is only known when the episode ends, up to 1200 steps later. It is also noisy: the same situation can end well or badly by chance.</p>
<p>We want to answer, <i>right now</i>: "how good is it to be here?" That answer is the <b>value</b>.</p>""",
read("same state, three runs of the same policy:\\n  G = 24.1   (drives the full 40 s)\\n  G = 18.7   (small wobble later)\\n  G =  9.3   (touches the line after 1.6 s)\\nvalue = the average over all such runs")))

b += sec("Value of a state, value of an action","definitions", split(
eq("V<sup>π</sup>(s)   = E[ G<sub>t</sub> | s<sub>t</sub> = s ]               state value\\nQ<sup>π</sup>(s,a) = E[ G<sub>t</sub> | s<sub>t</sub> = s, a<sub>t</sub> = a ]      action value")+
"""<p><b>V</b>: the return you expect from state s if you keep following policy π.</p>
<p><b>Q</b>: the same, but forcing action a first, then following π.</p>
<p>Both depend on π: the same position is worth a lot to a good driver and little to a bad one. When the policy improves, the values change too. That is why the critic is retrained at every iteration.</p>""",
fig("vstates","Three states, three values (illustrative, with your current maximum of 25).")+
"<p>The gap between Q and V is the core of the policy gradient (page 5): <b>A(s,a) = Q(s,a) − V(s)</b>, how much better action a is than the policy's usual choice.</p>"))

b += sec("The Bellman equation","derived from the return", split(
"""<p>Start from the recursive return of page 1:</p>"""+eq("G<sub>t</sub> = r<sub>t</sub> + γ·G<sub>t+1</sub>")+
"""<p>Take the expectation of both sides, given s<sub>t</sub> = s:</p>"""+eq("V(s) = E[ r<sub>t</sub> + γ·G<sub>t+1</sub> | s ]\\n     = E[ r<sub>t</sub> + γ·V(s<sub>t+1</sub>) | s ]")+
"""<p>The second line uses: the expected return from s<sub>t+1</sub> <i>is</i> V(s<sub>t+1</sub>). This is the <b>Bellman equation</b>:</p>
<div class='key'><b>Bellman.</b> The value here = the reward now + γ × the value of where you land.</div>
<p>It links each state's value to its neighbours'. It does not need the end of the episode: one step of experience is enough to check (and correct) a value estimate.</p>""",
fig("bell","A value is the average, over possible next states, of reward + γ·next value.")+
"""<p>At a <b>terminal</b> state (off_track) there is no next state: V(s′) = 0. At a <b>time-out</b> the next state exists, so V(s′) stays. This is the γ·V(s) that rsl_rl adds on time-outs (page 0).</p>"""))

b += sec("Break it: watch values spread","interactive", split(
"""<p>A toy version of your track: 11 lateral cells, 8 cm apart. The outer cells are terminal (a wheel on the line). The policy is bad on purpose: each step it drifts one cell left (25 %), stays (50 %) or drifts right (25 %). Reward per step = the centering bell, 1 at the centre.</p>
<p>Two ways to compute V:</p>
<ul><li><b>Bellman sweep</b>: apply the equation to every cell at once, using the known drift probabilities.</li>
<li><b>TD learning</b>: no probabilities given. Drive sampled episodes and correct V cell by cell from what happens (next section).</li></ul>"""+pred("pr2"),
widget("<label>method <select id='mth'><option value='sweep'>Bellman sweep</option><option value='td'>TD learning from samples</option></select></label>"+rng("gv","γ ",0.5,0.99,0.01,0.99)+rng("al","TD step size α ",0.01,0.6,0.01,0.1),
"<div id='v-tl'></div><div id='vb'></div><div class='read' id='vb-t'></div>"))+
"<p class='small'>Exact values for γ = 0.99 (verified): centre 29.83, next to the line 9.64. With γ = 0.9 the centre is only 8.28: the far future barely counts.</p>"+
key("Information travels one step per update: the line's danger reaches the centre only after many sweeps. In a real network, this is why the critic needs many iterations before its values are useful."))

b += sec("Monte Carlo vs Temporal Difference","two ways to learn V", split(
"""<p>We do not know the drift probabilities of the real car. We only have experience. Two ways to use it:</p>"""+
table(["","Monte Carlo (MC)","Temporal Difference (TD)"],[
["target for V(s)","the real return G<sub>t</sub>","r<sub>t</sub> + γ·V(s<sub>t+1</sub>)"],
["wait for","the end of the episode","one step"],
["bias","none (real outcome)","yes: uses the current, imperfect V"],
["noise (variance)","high: sums many random rewards","low: one random reward"],
["update","V ← V + α(G − V)","V ← V + α·δ"]])+
eq("δ<sub>t</sub> = r<sub>t</sub> + γ·V(s<sub>t+1</sub>) − V(s<sub>t</sub>)        the TD error")+
"""<p><b>δ is the surprise</b> of one step: positive if the step went better than V predicted.</p>""",
"""<p>Using a guess (V) to build the target for another guess is called <b>bootstrapping</b>. It is what lets TD learn while driving, before the episode ends.</p>
<p>The α slider above shows the trade-off on step size: α = 0.02 converges slowly but accurately (centre 29.2 after 3000 episodes), α = 0.5 jumps around the truth (36.6 at the same point).</p>"""+
why("not just pick one","MC is unbiased but noisy; TD is steady but biased while V is still wrong. GAE (page 6) blends them with one parameter, λ = 0.95 in your config. δ defined here is the building block of GAE.")))

b += sec("The critic in your project","V as a neural network", split(
"""<p>With continuous positions, speeds and angles, V cannot be a table. It is a network <b>V<sub>φ</sub>(s)</b> with parameters φ:</p>"""+
read("critic: 8 inputs → 128 → 64 → 32 → 1 output   (ELU activations, 11 521 weights)")+
"""<p>It is trained like a regression: after each data collection, compute a target return for every sample, then minimise</p>"""+eq("value loss = ( V<sub>φ</sub>(s) − target )²")+
"<p>That is <code>Loss/value</code> in TensorBoard. A value loss that keeps growing means the critic cannot keep up with the changing policy, or the rewards are badly scaled.</p>",
why("value_loss_coef = 1.0","The total loss adds the actor and critic terms (page 8). The coefficient balances them. With separate networks for actor and critic, as in your config, the gradients do not mix much, so 1.0 is a safe default.")+
why("the critic gets privileged inputs","Better inputs give better values, and better values give less noisy learning signals for the actor. The exact lateral error is far more informative than the camera estimate, and it costs nothing because the critic never runs on the real car.")+
anchor([(P+"mdp/observations.py:54","privileged_track_state: the critic's input"),(P+"config/f1_car/agents/rsl_rl_ppo_cfg.py","critic hidden_dims [128, 64, 32]"),(PPO+":154","time-outs: reward += γ·V(s), the bootstrap of a truncated episode")]))+
deep("why PPO learns V and not Q", """<p>Q(s,a) needs the action as an input and, to pick the best action, a maximisation over all actions. With continuous actions (steer, throttle) that maximisation is itself an optimisation problem. Methods like DDPG and SAC do learn Q, with an extra actor trained to maximise it.</p>
<p>PPO avoids this: it learns V(s) only, and estimates the advantage A = Q − V from real rewards plus V (page 6). The actor's job is then simply: make actions with positive advantage more likely.</p>
<p>The Bellman <i>optimality</i> equation, V*(s) = max<sub>a</sub> E[r + γV*(s′)], defines the value of the best policy. Q-learning methods iterate it. PPO instead evaluates its current policy (Bellman <i>expectation</i> equation) and improves the policy by gradient steps.</p>"""))

s2 = r'''
svg('vstates',640,150, box(10,20,190,80,['centred, aligned','V ≈ 24'],{col:C.good})+box(225,20,190,80,['drifting outward','V ≈ 12'],{col:C.mut})+box(440,20,190,80,['wheel 2 cm from line','V ≈ 2'],{col:C.acc})+T(320,130,'same policy π, three states',{col:C.mut,size:11}));
svg('bell',640,220, box(20,85,110,50,['s','V(s) = ?'])+L([[130,110],[300,40]],{arrow:true})+L([[130,110],[300,110]],{arrow:true})+L([[130,110],[300,180]],{arrow:true})+
 box(305,15,130,50,["s′₁","V(s′₁)"])+box(305,85,130,50,["s′₂","V(s′₂)"])+box(305,155,130,50,["s′₃ terminal","V = 0"],{col:C.acc})+
 T(540,44,'r₁ + γ·V(s′₁)',{col:C.good})+T(540,114,'r₂ + γ·V(s′₂)',{col:C.good})+T(540,184,'r₃ + 0',{col:C.acc})+T(220,210,'V(s) = probability-weighted average of the three',{size:11,col:C.mut}));
predict('pr2','start from V = 0 everywhere and apply one Bellman sweep. What is V at the centre cell?',['0','1.0','about 30'],1,
 'After one sweep every cell only knows its own reward: V(centre) = r = 1 + 0.99 × 0. It takes about 20 sweeps to reach 14.9 and 400 to reach the true 29.83: the future has to travel in, one step per sweep.');
const N=11, xs=[...Array(N)].map((_,i)=>(i-5)*0.08), r=xs.map(x=>Math.exp(-x*x/(2*.0625))), labels=xs.map(x=>(x*100).toFixed(0));
let hist=[], mode='sweep';
function build(v){
  hist=[]; let V=Array(N).fill(0); hist.push({V:V.slice(),note:'start: V = 0 everywhere'});
  if(v.mth==='sweep'){ for(let s=1;s<=400;s++){ V=V.map((_,i)=>(i==0||i==N-1)?0:r[i]+v.gv*(.25*V[i-1]+.5*V[i]+.25*V[i+1])); if(s<=30||s%10==0) hist.push({V:V.slice(),note:`after sweep ${s}`}); } }
  else { let ep=0; for(let k=0;k<=150;k++){ for(let e=0;e<20;e++){ ep++; let s=5; while(s>0&&s<N-1){ const u=Math.random(), s2=u<.25?s-1:u<.75?s:s+1; const tgt=r[s]+v.gv*((s2==0||s2==N-1)?0:V[s2]); V[s]+=v.al*(tgt-V[s]); s=s2; } } hist.push({V:V.slice(),note:`after ${ep} sampled episodes`}); } }
  let E=Array(N).fill(0); for(let it=0;it<20000;it++) E=E.map((_,i)=>(i==0||i==N-1)?0:r[i]+v.gv*(.25*E[i-1]+.5*E[i]+.25*E[i+1]));
  hist.exact=E;
}
const vt=timeline('v-tl',()=>hist.length,k=>{ const h=hist[k];
  bars('vb',h.V,{labels,max:Math.max(...hist.exact)*1.1,cols:h.V.map((_,i)=>(i==0||i==N-1)?C.acc:C.blue),xl:'lateral position of the car centre (cm)'});
  $('vb-t').textContent=`centre: ${h.V[5].toFixed(2)}  (exact ${hist.exact[5].toFixed(2)})     next to the line: ${h.V[1].toFixed(2)}  (exact ${hist.exact[1].toFixed(2)})`;
  return h.note; },{ms:90});
bind(['mth','gv','al'],v=>{ build(v); vt.reset(); });
'''
page("02-value","2 · Value, Bellman and TD","How good is it to be here? Learning to predict the return without waiting for it.",b,s2)

# ================= PAGE 3 : POLICY =================
b = sec("Why not just output a steering angle?","the problem", split(
"""<p class="lead">The simplest policy is a function: observation in, action out. Same input, always the same action. It is called <b>deterministic</b>.</p>
<p>For learning, it has two flaws:</p>
<ul><li><b>No exploration.</b> It only ever tries what it already does, so it can never discover that steering a bit earlier pays more.</li>
<li><b>No gradient through the world.</b> To improve, we must know how the return changes when the action changes. The simulator (physics, camera, detector) cannot be differentiated. A probability over actions can (page 4).</li></ul>""",
fig("detsto","Deterministic: one action. Stochastic: a probability over actions, sampled at each step.")))

b += sec("The Gaussian policy","what the actor really outputs", split(
"""<p>Your actor is a <b>stochastic policy</b> π<sub>θ</sub>(a|s): for each observation, a bell curve (Gaussian, or normal distribution) over each action.</p>"""+
eq("a ∼ N( μ<sub>θ</sub>(s), σ² )\\n\\nμ<sub>θ</sub>(s) : the mean, computed by the network from the observation\\nσ      : the standard deviation, one learned number per action")+
"""<p><b>θ</b> is the set of all actor weights (43 330 here). Training changes θ, which moves μ for every observation at once.</p>
<p>In your config σ is <b>state-independent</b>: the same spread whatever the car sees. It is a separate parameter, not a network output.</p>
<p>In play mode the mean is used directly: no sampling, no noise.</p>""",
fig("flow","Mean from the network, std from one learned parameter per action, then one random draw.")+
why("a Gaussian","Continuous actions need a continuous distribution. The Gaussian has a closed-form log-probability and entropy (both needed by PPO), one knob for 'where' (μ) and one for 'how wide' (σ), and is cheap to sample on the GPU for 64 cars at once.")+
anchor([("rsl_rl/modules/distribution.py:166","std_param = init_std × ones(output_dim): σ is a plain learned parameter"),("rsl_rl/modules/distribution.py:191","Normal(mean, std): the distribution sampled each step"),("rsl_rl/modules/distribution.py:197","deterministic_output = the mean (play mode)")])))

b += sec("Log-probability: the quantity PPO pushes on","formula, then meaning", split(
"""<p>The density of one action value a under N(μ, σ²):</p>"""+eq("π(a|s) = 1/(σ√(2π)) · exp( −(a − μ)² / (2σ²) )")+
"""<p>Its logarithm, summed over the two actions (steer and throttle are independent):</p>"""+eq("log π(a|s) = Σ<sub>i</sub> [ −(a<sub>i</sub> − μ<sub>i</sub>)² / (2σ<sub>i</sub>²) − log σ<sub>i</sub> − ½·log 2π ]")+
"""<p>Why the log: products of probabilities over many steps become sums, numbers stay in a safe range, and the gradient is simple:</p>"""+eq("∂ log π / ∂μ = (a − μ) / σ²"),
"""<p>Read that last line: if the sampled action a was above the mean, increasing log π means <b>moving μ toward a</b>. The step is larger when σ is small.</p>
<p>That is the whole mechanism of policy learning: after an action turned out well, move the mean toward it; after it turned out badly, move away. Page 4 derives it.</p>"""))

b += sec("Break it: sampling and the ±1 wall","interactive", split(pred("pr3")+
"""<p>Actions are clamped to −1…1 by the environment <i>after</i> sampling. Push the mean near the edge and watch the samples pile up on the wall.</p>
<h3>Why the limit exists</h3>
<p>−1…1 maps to physical limits: steer ±0.45 rad (the knuckle's range), throttle 0…3 m/s. The network has no idea of these limits, so the environment enforces them.</p>
<h3>The hidden cost</h3>
<p>rsl_rl computes log π on the <b>raw</b> sample (e.g. 1.3), while the car executed 1.0. All samples beyond the wall have the same effect on the car but different probabilities. The mean can drift past ±1 with no visible change, and the gradient toward it comes from noise. A policy whose mean sits beyond the wall has effectively saturated.</p>""",
widget(rng("mu","mean μ ",-1.5,1.5,0.01,0.9)+rng("sd","std σ ",0.02,1,0.01,0.2)+"<button id='go'>sample 400 actions</button>","<div id='bell'></div><div class='read' id='bell-t'></div>")+
"<p class='small'>Verified: μ = 0.9, σ = 0.2 → 30.9 % clipped. μ = 1.2 → 84.1 %. σ = 1 (rsl_rl's default init_std) at μ = 0 → 31.7 %.</p>")+
why("clip_actions is null in your agent config","Isaac Lab's wrapper can clip actions before they reach the environment; with <code>clip_actions: null</code> it does not, and the clamp happens in your action term. Either way the policy samples unbounded values. The common alternative is a tanh 'squashing' of the sample, with a log-probability correction, as in SAC.")+
anchor([(P+"mdp/actions.py:92","torch.clamp(actions, −1, 1)"),("isaaclab_rl/rsl_rl/vecenv_wrapper.py:157","optional clip, disabled (clip_actions = None)")]))

b += sec("The action is not what the car does","actuator dynamics", split(
"""<p>Between the policy's action and the wheels, your action term adds two filters to the steering:</p>"""+
eq("target   = a<sub>steer</sub> × 0.45 rad\\nfiltered = 0.7·current + 0.3·target          (low-pass, α = 0.3)\\ncurrent += clamp(filtered − current, ±0.1)    (slew limit 3 rad/s × 1/30 s)")+
"""<p>So a full-lock command needs several steps to take effect. Predict, then step through.</p>"""+pred("pr4"),
widget("<label>command <select id='cmd'><option value='lock'>0 → full right (0.45 rad)</option><option value='rev'>full left → full right</option></select></label>","<div id='st-tl'></div><div id='st'></div>"))+
why("filter the steering at all","A real servo cannot jump; a policy sampled with noise σ would otherwise make the wheels shake at 30 Hz. The filters give smooth, realistic steering. The cost is <b>lag</b>: a full reversal needs 12 steps (0.4 s) to reach 90 %. The policy must anticipate, and the return must look at least that far ahead (γ = 0.99 does).")+
why("steer is in the observation","The filter has memory (the current angle). Without seeing it, the policy could not know whether its last command has arrived yet: the observation would not be Markov (page 0).")+
anchor([(P+"mdp/actions.py:100","EMA low-pass, steer_filter_alpha = 0.30"),(P+"mdp/actions.py:103","slew limit max_steer_rate = 3.0 rad/s"),(P+"mdp/actions.py:111","throttle rate limit 3.0 per second"),(P+"mdp/observations.py:126","steer angle read back into the observation")]))

b += sec("Entropy: measuring how random the policy is","exploration", split(
"""<p><b>Entropy</b> H measures the spread of a distribution. For a Gaussian it depends only on σ:</p>"""+eq("H = ½·log(2πe) + log σ        per action")+
table(["σ","H per action","H for 2 actions (logged)"],[["1.0","1.42","2.84"],["0.2 (your init)","−0.19","−0.38"],["0.05","−1.58","−3.15"],["0.019 (your run, end)","−2.54","−5.09"]])+
"""<p>Negative values are normal: for continuous distributions, entropy is relative. Only its change matters.</p>""",
"""<p>PPO adds a small <b>entropy bonus</b> to the loss (entropy_coef = 0.008). It rewards keeping σ larger, which fights premature collapse: a policy that stops exploring before it has found the good behaviour stays stuck.</p>
<p>In TensorBoard, <code>Loss/entropy</code> and <code>Policy/mean_std</code> tell the same story through H = const + log σ.</p>"""+
why("init_std = 0.2 and not 1.0","rsl_rl's default is 1.0. In your units, σ = 1 means steering noise of ±0.45 rad (full lock either way) and a third of all actions hitting the wall: the car would leave the track before learning anything. σ = 0.2 means ±0.09 rad of steering noise and ±0.3 m/s of speed noise: enough to explore, small enough to survive.")+
why("there is no lower or upper limit on σ","rsl_rl's default std_range is (1e-6, 1e6): effectively unbounded. Nothing but the entropy bonus stops σ from collapsing. In your run it fell from 0.2 to 0.019 once the task was solved, which is fine then, but harmful if it happens before success.")+
anchor([("rsl_rl/modules/distribution.py:148","std_range = (1e-6, 1e6)"),(P+"config/f1_car/agents/rsl_rl_ppo_cfg.py:22","init_std = 0.2"),(P+"config/f1_car/agents/rsl_rl_ppo_cfg.py:33","entropy_coef = 0.008")]))+
deep("other ways to parameterise the policy", """<ul><li><b>State-dependent σ</b>: the network also outputs σ(s), so the policy can be confident on straights and exploratory in bends. More expressive, less stable.</li>
<li><b>log σ instead of σ</b> (<code>std_type: "log"</code>): the parameter is log σ, so σ can never become negative and changes are relative. rsl_rl supports both.</li>
<li><b>Squashed Gaussian</b>: a = tanh(u), u ∼ N(μ, σ²). Actions stay inside ±1 by construction; log π gets a correction term −Σ log(1 − tanh²u).</li>
<li><b>Beta distribution</b>: naturally bounded on [0, 1], used in some driving work.</li></ul>"""))

s3 = r'''
svg('detsto',640,190, box(10,20,140,50,['observation']) + L([[150,45],[196,45]],{arrow:true}) + box(200,20,140,50,['deterministic']) + L([[340,45],[386,45]],{arrow:true}) + box(390,20,240,50,['always steer = 0.31'],{col:C.mut})+
 box(10,110,140,50,['observation']) + L([[150,135],[196,135]],{arrow:true}) + box(200,110,140,50,['stochastic'],{col:C.acc}) + L([[340,135],[386,135]],{arrow:true}) + box(390,110,240,50,['0.27, 0.35, 0.30, 0.22, …','new draw every step'],{col:C.acc}));
svg('flow',640,150, box(10,50,110,50,['observation','7 numbers'])+L([[120,75],[156,75]],{arrow:true})+box(160,50,120,50,['actor MLP','256-128-64'],{col:C.acc})+L([[280,75],[316,75]],{arrow:true})+box(320,10,120,50,['mean μ','2 numbers'])+box(320,90,120,50,['std σ','2 parameters'],{col:C.mut})+
 L([[440,35],[496,65]],{arrow:true})+L([[440,115],[496,85]],{arrow:true})+box(500,50,130,50,['sample a','~ N(μ, σ²)'],{col:C.blue}));
predict('pr3','mean 0.9, std 0.2. What share of the sampled actions hits the +1 wall?',['about 3 %','about 30 %','about 70 %'],1,'30.9 %: the wall is only half a standard deviation above the mean, and everything beyond it is executed as exactly 1.0.');
function erf(x){const t=1/(1+0.3275911*Math.abs(x));const y=1-(((((1.061405429*t-1.453152027)*t)+1.421413741)*t-0.284496736)*t+0.254829592)*t*Math.exp(-x*x);return x>=0?y:-y;}
const Phi=x=>0.5*(1+erf(x/Math.SQRT2));
function drawBell(v,samples){
 const pdf=x=>Math.exp(-0.5*((x-v.mu)/v.sd)**2)/(v.sd*Math.sqrt(2*Math.PI)); const p=[]; for(let x=-2;x<=2;x+=0.005) p.push([x,pdf(x)]);
 const ymax=Math.max(pdf(v.mu),1)*1.15;
 chart('bell',[{pts:p,col:C.acc,label:'density of the raw sample'}],{xmin:-2,xmax:2,ymin:0,ymax,xl:'steer action',vlines:[{x:-1,label:'−1 wall',col:C.acc},{x:1,label:'+1 wall',col:C.acc}]});
 const clipped=100*(1-Phi((1-v.mu)/v.sd)+Phi((-1-v.mu)/v.sd));
 let txt=`clipped (theory): ${clipped.toFixed(1)} %     steer noise: ±${(v.sd*0.45).toFixed(3)} rad`;
 if(samples){ const W=720,m=58, X=a=>m+(a+2)/4*(W-m-14); let s='', hit=0; const bins={};
   samples.forEach(a=>{ const c=Math.max(-1,Math.min(1,a)); if(c!==a) hit++; const b=Math.round(c*40); bins[b]=(bins[b]||0)+1; });
   Object.entries(bins).forEach(([b,n])=>{ const x=X(b/40); s+=`<rect x="${x-3}" y="${216-Math.min(n,160)}" width="6" height="${Math.min(n,160)}" fill="${Math.abs(b)==40?C.acc:C.blue}" opacity=".55"/>`; });
   $('bell').querySelector('svg').insertAdjacentHTML('beforeend',s); txt+=`\nsampled: ${hit} of ${samples.length} clipped (${(100*hit/samples.length).toFixed(1)} %). Blue bars: what the car executes. Orange bar at the wall: every clipped sample.`; }
 $('bell-t').textContent=txt;
}
bind(['mu','sd'],v=>drawBell(v));
$('go').onclick=()=>{ const v={mu:+$('mu').value,sd:+$('sd').value}; drawBell(v,[...Array(400)].map(()=>v.mu+v.sd*randn())); };
predict('pr4','from straight to full right lock, how many decisions until the wheels reach 90 % of 0.45 rad?',['1 (instant)','about 3','about 7'],2,
 '7 steps (0.23 s): the first steps are capped at 0.1 rad each by the slew limit, then the low-pass filter slows the end. A full left-to-right reversal takes 12 steps (0.4 s) to reach 90 %.');
let trace=[];
function steer(from){ let cur=from; const p=[[0,cur]]; for(let k=1;k<=24;k++){ const f=0.7*cur+0.3*0.45; cur+=Math.max(-0.1,Math.min(0.1,f-cur)); p.push([k,cur]); } return p; }
const stl=timeline('st-tl',()=>25,k=>{ const cmd=trace.map(([t])=>[t,0.45]);
 chart('st',[{pts:cmd,col:C.mut,dash:true,label:'command'},{pts:trace.slice(0,k+1),col:C.acc,dots:true,label:'actual steering angle'}],{xmin:0,xmax:24,ymin:-0.5,ymax:0.5,xl:'decision steps (1/30 s)',yl:'rad',vlines:[{x:7,label:'7'},{x:12,label:'12'}]});
 const c=trace[k][1]; return `angle ${c.toFixed(3)} rad = ${(100*(c-trace[0][1])/(0.45-trace[0][1])).toFixed(0)} % of the way`; },{ms:120});
bind(['cmd'],v=>{ trace=steer(v.cmd==='lock'?0:-0.45); stl.reset(); });
'''
page("03-policy","3 · The stochastic policy","What the actor really outputs: a bell curve over actions, and what happens between that curve and the wheels.",b,s3)

# ================= PAGE 4 : POLICY GRADIENT =================
b = sec("We cannot differentiate the world","the problem", split(
"""<p class="lead">Goal: change the actor weights θ so that the expected return J(θ) goes up. Gradient ascent needs ∂J/∂θ.</p>
<p>But the chain θ → action → <i>physics, camera, detector</i> → reward goes through a simulator we cannot differentiate. PhysX gives no gradients; even if it did, contacts make them meaningless.</p>
<p>The <b>policy gradient</b> gets the gradient anyway, without touching the simulator. The trick: differentiate the <i>probability</i> of the actions, which we own, not the world.</p>""",
fig("chain","The gradient path. Orange: no gradient exists. Green: the only part we can differentiate.")))

b += sec("The log-derivative trick","derivation, one line at a time", split(
"""<p>Start with a single decision (a 'bandit'): one action a ∼ π<sub>θ</sub>, one reward R(a).</p>"""+
eq("J(θ) = E<sub>a∼π<sub>θ</sub></sub>[ R(a) ] = ∫ π<sub>θ</sub>(a) R(a) da\\n\\n∇J = ∫ ∇π<sub>θ</sub>(a) R(a) da                    R does not depend on θ\\n   = ∫ π<sub>θ</sub>(a) ∇log π<sub>θ</sub>(a) R(a) da        because ∇π = π·∇log π\\n   = E<sub>a∼π<sub>θ</sub></sub>[ R(a) · ∇log π<sub>θ</sub>(a) ]")+
"""<p>The last line is an <b>expectation</b>, so we can estimate it by sampling: try N actions, then average R·∇log π.</p>""",
"""<h3>What it means</h3>
<p>For each sampled action, ∇log π points in the direction that makes that action <i>more likely</i>. Weight it by how good the action was. Good actions pull the policy toward themselves, bad ones push it away.</p>
<p>With the Gaussian of page 3, ∇<sub>μ</sub> log π = (a − μ)/σ². So the update of the mean is:</p>"""+eq("Δμ ∝ average of  R(a) · (a − μ) / σ²")+
"<p>No physics equation appears anywhere. The simulator only has to return R.</p>"+
key("Policy gradient = try actions, score them, make the well-scored ones more likely. It works through any black-box environment.")))

b += sec("Break it: a one-step steering bandit","interactive", split(
"""<p>One decision: choose a steer a. The ideal is a* = 0.1, reward R = −(a − a*)². The policy starts with μ = 0.6. Each iteration samples a batch, estimates the gradient, and moves μ.</p>
<p>True gradient at μ = 0.6: −2(μ − a*) = −1.0, so μ should go down.</p>"""+pred("pr5")+
"""<p>Things to try:</p><ul><li>batch 1: μ wanders, sometimes uphill</li><li>σ very small: steps explode (the 1/σ² factor)</li><li>large learning rate: overshoot</li></ul>""",
widget(rng("bs","batch ",1,256,1,8)+rng("lr","learning rate ",0.001,0.1,0.001,0.02)+rng("sg","σ ",0.02,0.8,0.01,0.2),
"<div id='pg-tl'></div><div id='pg'></div><div id='pg2'></div><div class='read' id='pg-t'></div>"))+
table(["Batch size","Mean gradient estimate","Std of the estimate","Points the wrong way"],[["1","−0.97","2.37","51 %"],["8","−1.01","0.84","6.6 %"],["64","−1.00","0.30","0 %"],["1024","−1.00","0.07","0 %"]])+
"<p class='small'>Verified with 4000 repeats at μ = 0.6, σ = 0.2. The estimate is unbiased (right on average) but with batch 1 it is a coin flip.</p>")

b += sec("From one action to a whole trajectory","the real case", split(
"""<p>A trajectory τ = (s₀, a₀, s₁, a₁, …) has probability</p>"""+eq("p<sub>θ</sub>(τ) = p(s₀) · Π<sub>t</sub> π<sub>θ</sub>(a<sub>t</sub>|s<sub>t</sub>) · P(s<sub>t+1</sub>|s<sub>t</sub>, a<sub>t</sub>)")+
"""<p>Take the log: products become sums. The dynamics terms P do not depend on θ, so their gradient is zero. Only the policy terms remain:</p>"""+
eq("∇log p<sub>θ</sub>(τ) = Σ<sub>t</sub> ∇log π<sub>θ</sub>(a<sub>t</sub>|s<sub>t</sub>)")+
"<p>This is why the unknown physics drops out: it is a fixed factor in the probability, independent of θ.</p>",
eq("∇J = E[ Σ<sub>t</sub> ∇log π<sub>θ</sub>(a<sub>t</sub>|s<sub>t</sub>) · G<sub>t</sub> ]")+
"""<p>Each action is weighted by the return <b>from its own step onward</b>, G<sub>t</sub>, not the episode's total. An action cannot change rewards that came before it (<b>causality</b>), so including them only adds noise.</p>
<p>This estimator is called <b>REINFORCE</b> (Williams, 1992). PPO is REINFORCE plus three fixes for its noise and instability.</p>"""))

b += sec("Why plain REINFORCE is not enough","the three weaknesses", split(
table(["Weakness","Why","Fix, later in this course"],[
["<b>very noisy</b>","G<sub>t</sub> sums hundreds of random rewards; every action in a good episode gets credit, even the bad ones","baseline (page 5), GAE (page 6)"],
["<b>scale-dependent</b>","if all returns are positive (yours are), every action is pushed up, just by different amounts","advantage A = G − V (page 5)"],
["<b>one step per batch</b>","after one update, the data comes from an older policy, so reusing it is not allowed by the derivation","importance ratio + clip (pages 7–8)"]]),
"""<p><b>The σ effect.</b> The gradient estimate's noise is not monotonic in σ (batch 16, verified):</p>"""+
table(["σ","Std of the estimate"],[["0.05","1.32"],["0.2","0.60"],["0.6","0.77"]])+
"<p>Too small: 1/σ² amplifies every deviation. Too large: rewards of wildly different actions mix. Your init_std = 0.2 sits near the sweet spot of this toy problem.</p>"))

b += sec("In your project","where REINFORCE hides inside PPO", split(
"""<p>rsl_rl never writes 'REINFORCE'. But look at its loss at the very first gradient step of an iteration, when the new policy still equals the old one (ratio = 1):</p>"""+
eq("surrogate = −A · ratio ,  ratio = π<sub>θ</sub>(a|s) / π<sub>old</sub>(a|s)\\n\\n∇ratio at θ = θ<sub>old</sub>  =  ∇log π<sub>θ</sub>(a|s)\\n⇒ ∇surrogate = −A · ∇log π<sub>θ</sub>(a|s)")+
"<p>Exactly the policy gradient, with the advantage A in place of G<sub>t</sub>, and a minus sign because optimisers minimise.</p>",
anchor([(PPO+":269","ratio = exp(log π_new − log π_old)"),(PPO+":270","surrogate = −advantages × ratio"),("rsl_rl/modules/distribution.py:230","log_prob summed over steer and throttle")])+
key("Every PPO update starts as a policy-gradient step. Everything else in PPO controls how far that step may go.")))

b += deep("why ∇π = π·∇log π, and why E[∇log π] = 0", """<p>From calculus: ∇log f = ∇f / f, so ∇f = f·∇log f. Applied to π<sub>θ</sub>(a) this gives the trick.</p>
<p>A useful identity follows: E<sub>a∼π</sub>[∇log π(a)] = ∫ ∇π(a) da = ∇∫ π(a) da = ∇1 = 0. The score has zero mean. So subtracting any constant b from R does not change the expected gradient:</p>"""+
eq("E[ (R − b)·∇log π ] = E[ R·∇log π ] − b·0")+
"""<p>…but it can strongly reduce the variance. With b = E[R], the batch-1 estimate in the toy problem drops from std 2.37 to 1.75, and from 51 % to 7 % wrong-sign. That is the <b>baseline</b>, page 5, and V(s) is the natural choice for b.</p>""")

s4 = r'''
svg('chain',640,140, box(10,45,90,50,['θ','weights'],{col:C.good})+L([[100,70],[136,70]],{arrow:true,col:C.good})+box(140,45,110,50,['π_θ(a|s)','action prob.'],{col:C.good})+L([[250,70],[286,70]],{arrow:true,col:C.acc})+
 box(290,45,200,50,['physics, camera, detector','black box'],{col:C.acc})+L([[490,70],[526,70]],{arrow:true,col:C.acc})+box(530,45,100,50,['reward R'])+
 T(195,30,'differentiable',{col:C.good,size:11})+T(390,30,'no gradient',{col:C.acc,size:11})+T(320,125,'policy gradient: only the green part is differentiated',{size:11,col:C.mut}));
predict('pr5','with batch 1, in what share of iterations does the gradient estimate point the WRONG way (μ moves up, away from 0.1)?',['about 5 %','about 25 %','about 50 %'],2,
 '51 %: a single sample is almost a coin flip, though the average over many is exactly right. With batch 8 it drops to 6.6 %, with 64 to 0 %. Your training uses 4096 samples per iteration for this reason.');
const astar=0.1; let run=[];
function simulate(v){ run=[]; let mu=0.6; for(let it=0;it<=120;it++){ const s=[]; let g=0; for(let j=0;j<v.bs;j++){ const a=mu+v.sg*randn(); s.push(a); g+=-((a-astar)**2)*(a-mu)/(v.sg*v.sg);} g/=v.bs; run.push({mu,g,s}); mu=mu+v.lr*g; mu=Math.max(-3,Math.min(3,mu)); } }
const pg=timeline('pg-tl',()=>run.length,k=>{ const h=run[k];
 chart('pg',[{pts:run.slice(0,k+1).map((r,i)=>[i,r.mu]),col:C.acc,label:'mean μ'},{pts:[[0,astar],[120,astar]],col:C.good,dash:true,label:'best a* = 0.1'}],{xmin:0,xmax:120,ymin:-0.6,ymax:1.2,xl:'iteration',h:220});
 const W=720,H=120,X=a=>58+(a+1)/2.5*(W-72); let s=L([[58,70],[W-14,70]],{col:C.line})+T(X(astar),100,'a*',{col:C.good})+`<line x1="${X(astar)}" x2="${X(astar)}" y1="40" y2="90" stroke="${C.good}" stroke-dasharray="4 3"/>`+`<line x1="${X(h.mu)}" x2="${X(h.mu)}" y1="35" y2="95" stroke="${C.acc}" stroke-width="2"/>`+T(X(h.mu),28,'μ',{col:C.acc,b:true});
 h.s.slice(0,64).forEach(a=>{ const R=-((a-astar)**2); s+=`<circle cx="${X(Math.max(-1,Math.min(1.5,a)))}" cy="70" r="4" fill="${C.blue}" opacity="${Math.max(.15,1+R*3)}"/>`; });
 const gx=Math.max(-150,Math.min(150,h.g*60)); s+=L([[X(h.mu),112],[X(h.mu)+gx,112]],{col:h.g<0?C.good:C.acc,sw:3,arrow:true});
 for(const t of [-1,-0.5,0,0.5,1,1.5]) s+=T(X(t),H-2,String(t),{size:10,col:C.mut});
 svg('pg2',W,H,s);
 const wrong=run.slice(0,k+1).filter(r=>r.g>0&&r.mu>astar).length;
 $('pg-t').textContent=`iteration ${k}: μ = ${h.mu.toFixed(3)}   gradient estimate ${h.g.toFixed(2)} (true ${(-2*(h.mu-astar)).toFixed(2)})\nwrong-way steps so far: ${wrong} of ${k+1}.  Dots: samples (darker = better reward). Arrow: estimated step direction.`;
 return 'batch '+$('bs').value; },{ms:90});
bind(['bs','lr','sg'],v=>{ simulate(v); pg.reset(); });
'''
page("04-gradient","4 · The policy gradient","How to improve a policy through a simulator you cannot differentiate.",b,s4)
