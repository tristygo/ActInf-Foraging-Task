# Two packages are required to run this script:
# pip install equinox
# pip install inferactively-pymdp

import numpy as np
import jax
import jax.numpy as jnp
from equinox import tree_at
from pymdp import utils
from pymdp.agent import Agent
import collections

################################################################################################################################################################
# Parameters

SEED = 2026
np.random.seed(SEED)
jax_key = jax.random.PRNGKey(SEED) 

T        = 5
max_time = 5000             # maximum number of time steps to run simulation
richness_dist=np.array([    # {rich, medium, sparse} (patch type)
    0.3,
    0.4,
    0.3
])
resource_dist=np.array([    # {rich, medium, sparse} -> {full, high, med, low, none} (initial resources)
    [0.40, 0.00, 0.00],
    [0.50, 0.20, 0.00],
    [0.10, 0.60, 0.10],
    [0.00, 0.20, 0.70],
    [0.00, 0.00, 0.20]
])

depletion_rate=0.20         # -20% (avg) on each forage step

num_states = [4, 5, T+1]    # factor 0: 4 patch_types: {rich, medium, sparse, travelling}
                            # factor 1: 5 resource_levels: {full, high, med, low, none}
                            # factor 2: T+1 locations {on_patch, travel_1, ... , travel_T}

num_obs = [4, 4, T+1]       # modality 0: 4 arrival cues: {promising, neutral, poor, travelling}
                            # modality 1: 4 food obs: {lots, some, little, none}
                            # modality 2: T+1 locations {on_patch, travel_1, ... , travel_T}

num_actions = [2]           # one control factor: stay / leave

################################################################################################################################################################
# A matrices

A = [None] * 3

A[0] = np.array([
    [0.90, 0.10, 0.00, 0.00],   # Promising
    [0.10, 0.80, 0.10, 0.00],   # Neutral
    [0.00, 0.10, 0.90, 0.00],   # Poor
    [0.00, 0.00, 0.00, 1.00],   # Travelling
])

A[1] = np.array([
    [1.00, 0.80, 0.10, 0.00, 0.00],   # Lots
    [0.00, 0.20, 0.80, 0.10, 0.00],   # Some
    [0.00, 0.00, 0.10, 0.80, 0.00],   # Little
    [0.00, 0.00, 0.00, 0.10, 1.00],   # None
])

A[2] = np.eye(T + 1)   # agent always knows its location

A_true = [a.copy() for a in A]   # ground-truth copy kept for environment

################################################################################################################################################################
# B matrices

B = [None] * 3

# Factor 0: Patch type
B[0] = np.zeros((4, 4, 2))
B[0][:, :, 0] = np.eye(4)       # stay: patch type persists
B[0][:, :, 1] = np.array([      # leave: becomes "travelling"
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [1, 1, 1, 1],
])

# Factor 1: Resource levels
B[1] = np.zeros((5, 5, 2))
# B[1][:,:,0] is unknown, depends on continuous resource depletion in env.py

B[1][:, :, 1] = np.array([      # leave: resources collapse to "none"
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1],
])

# Factor 2: Location
B[2] = np.zeros((T + 1, T + 1, 2))

row1 = np.concatenate(([1], np.zeros(T - 1), [1]))
row2 = np.zeros(T + 1)
row3 = np.hstack((np.zeros((T - 1, 1)), np.eye(T - 1), np.zeros((T - 1, 1))))
B[2][:, :, 0] = np.vstack((row1, row2, row3))      # stay: advance travel counter

row1 = np.concatenate((np.zeros(T), [1]))
row2 = np.hstack((np.eye(T), np.zeros((T, 1))))
B[2][:, :, 1] = np.vstack((row1, row2))            # leave: advance travel counter

################################################################################################################################################################
# C matrices (log-preferences)

C = [None] * 3

C[0] = np.array([0.0, 0.0, 0.0, 0.0])                # arrival cues
C[1] = np.array([5.0, 3.0, 1.0, -4.0])               # food observations
C[2] = np.concatenate(([0], np.repeat(-1, T)))       # include additional travel cost to prevent overstaying
C[2][T] = 4                                          # add arrival bonus to represent new patch opportunity and offset travel cost

################################################################################################################################################################
# D matrices (priors)

D = [None] * 3

D[0] = np.array([0.3, 0.4, 0.3, 0.0])             # patch type: learned from environment
D[1] = np.array([0.2, 0.2, 0.2, 0.2, 0.2])        # resource level: unknown, depends on patch type and initial resource distributions
D[2] = np.concatenate(([1], np.zeros(T)))         # always start on patch

D_true = [d.copy() for d in D]                    # keep copy of true D

################################################################################################################################################################
# Dirichlet concentration arrays (pB, pD)
# pD is maintained *outside* the agent (Agent has no pD field in v1.0.2) and used to drive manual D updates via tree_at after each episode.

pB = [None] * 3

pB[0] = None                                       # set after D-informed B[0] columns below

pB[1] = np.zeros((5, 5, 2))
pB[1][:, :, 0] = np.array([                        # stay: biased toward non-recovery
    [10, 1,  1,  1,  1],
    [20, 10, 1,  1,  1],
    [10, 20, 10, 1,  1],
    [1,  10, 20, 15, 1],
    [1,  1,  10, 25, 40],
])
pB[1][:, :, 1] = np.array([                        # leave: pinned to resource collapse
    [1e-6, 1e-6, 1e-6, 1e-6, 1e-6],
    [1e-6, 1e-6, 1e-6, 1e-6, 1e-6],
    [1e-6, 1e-6, 1e-6, 1e-6, 1e-6],
    [1e-6, 1e-6, 1e-6, 1e-6, 1e-6],
    [1e6,  1e6,  1e6,  1e6,  1e6 ],
])

pB[2] = B[2] * 1e6                                 # pinned: location dynamics are known

# External pD concentration arrays — updated manually each episode.
# D[2] is pinned (agent always starts on-patch) so only D[0] and D[1] are learned.
pD = [None] * 3
pD[0] = np.array([1., 1., 1., 1e-6])     # uniform over {rich, med, sparse}; travelling pinned near 0
pD[1] = np.ones(5)                       # uniform over resource levels
pD[2] = None                             # not learned

################################################################################################################################################################
# Initialise learned matrices from Dirichlet means

B[1][:, :, :] = utils.norm_dist(pB[1])
D[0]          = utils.norm_dist(pD[0])   # uniform over non-travel types
D[1]          = utils.norm_dist(pD[1])

# Travelling arrival column of B[0]: initialised to D[0] prior and kept in sync
# with D[0] as it is learned during the loop (via tree_at).
B[0][:, 3, 0] = D[0]
B[0][:, 3, 1] = D[0]

pB[0] = B[0] * 1e6

print("Matrices constructed")

################################################################################################################################################################
# Agent

agent = Agent(
    A=A,  A_dependencies=[[0], [1], [2]],
    B=B,  pB=pB,  num_controls=[2],
    B_action_dependencies=[[0], [0], [0]],
    C=C,
    D=D,
    learn_A=False,
    learn_B=True,
    learn_D=True,                       # doesn't actually do anything. Thanks pymdp.
    policy_len=T+3,                     # forward length of policy calc rollout
    inference_algo="fpi",               # calculate VFE w fixed point iteration
    gamma=8.0,                          # policy precision: higher => more decisive over policies
    alpha=8.0,                          # action precision: higher => more decisive over action sampling
    action_selection="stochastic",      # standard action sampling
    sampling_mode="full",
    use_states_info_gain=True,          # epistemic actions to learn hidden states
    use_param_info_gain=False,          # take actions to better learn pA, pB, pD ***
)

print("Agent defined")

################################################################################################################################################################
# KL Divergence Helper

def kl_div(p, q, eps=1e-10):
    """KL divergence KL(p || q), with small epsilon for numerical stability."""
    p = np.asarray(p, dtype=float) + eps
    q = np.asarray(q, dtype=float) + eps
    p = p / p.sum()
    q = q / q.sum()
    return float(np.sum(p * np.log(p / q)))

# True distributions to compare against
true_D0 = D_true[0].copy()      # richness_dist + travelling=0
true_A1 = A_true[1].copy()      # ground-truth food likelihood (4 x 5)

################################################################################################################################################################
# Main loop

from env import PatchWorld

world   = PatchWorld(A_true, T, richness_dist, resource_dist, depletion_rate)
episode = 0
step    = 0
log     = collections.defaultdict(list)

print("Beginning main loop")

while step < max_time:

    #### BEGIN EPISODE ####
    world.generate_patch()

    # Reset prior to D at every episode boundary.
    prior = list(agent.D)

    obs_history    = []   # list[t] of [cue_int, food_int, loc_int]
    qs_history     = []   # list[t] of qs — each is list[num_factors] of (1, 1, Ns_f)
    action_history = []   # list[t] of (1, num_flat_controls)
    harvest_history= []   # list[t] of harvest amounts

    t    = 0
    left = False

    #### FORAGING LOOP ####
    while not left:

        # 1. Observe
        obs = world.forage()                            # [cue_int, food_int, 0]
        harvest_history.append(world.last_harvest())

        # 2. Infer hidden states
        obs_batched = [jnp.array([o]) for o in obs]
        qs = agent.infer_states(obs_batched, prior)

        qs_history.append(qs)
        obs_history.append(obs)

        # 3. Policy inference
        q_pi, neg_efe_full = agent.infer_policies(qs)

        # 4. Action selection
        # Pass subkey for stochastic action sampling
        jax_key, subkey = jax.random.split(jax_key)
        action = agent.sample_action(q_pi, rng_key=subkey[None])
        action_history.append(action)

        t += 1

        # 5. Leave or propagate prior to next step
        if int(action[0, 0]) == 1 or t > 25:
            left = True
        else:
            # update_empirical_prior rolls beliefs forward through B
            prior = agent.update_empirical_prior(action, qs)

    #### TRAVEL PHASE ####
    # Propagate prior through the leave action to get the travel start state.
    # Use action=0 (stay) as the dummy travel action, per B[2][:,:,0]
    travel_prior  = agent.update_empirical_prior(action, qs)
    dummy_action  = jnp.zeros_like(action)

    for travel_step in range(1, T + 1):
        t_obs         = world.travel(travel_step)           # [3, 3, travel_step]
        t_obs_batched = [jnp.array([o]) for o in t_obs]

        t_qs = agent.infer_states(t_obs_batched, travel_prior)

        obs_history.append(t_obs)
        qs_history.append(t_qs)
        action_history.append(dummy_action)
        harvest_history.append(0) # no harvest while travelling

        if travel_step < T:
            # Update prior for next travel step
            travel_prior = agent.update_empirical_prior(dummy_action, t_qs)

    #### LEARNING (once per episode) ####
    # B learning only uses foraging steps: all travel dynamics known
    all_beliefs = [
        jnp.stack(
            [qs_history[i][f][:, -1] for i in range(t)],
            axis=1
        )
        for f in range(agent.num_factors)
    ]

    all_obs = [
        jnp.array([[obs_history[i][m] for i in range(t)]])
        for m in range(agent.num_modalities)
    ]

    all_actions = jnp.stack(action_history[:t], axis=1)

    agent = agent.infer_parameters(
        beliefs_A=all_beliefs, # we don't learn A, but pymdp requires us to pass this in regardless
        observations=all_obs,
        actions=all_actions,
        beliefs_B=all_beliefs,
    )

    # D learning — manual Dirichlet update using first-timestep beliefs.
    qs_t0 = [np.array(qs_history[0][f][0, -1]) for f in range(agent.num_factors)]

    pD[0] += qs_t0[0]           # accumulate patch-type evidence
    pD[1] += qs_t0[1]           # accumulate initial-resource evidence

    new_D0 = jnp.array(pD[0] / pD[0].sum())   # (4,)
    new_D1 = jnp.array(pD[1] / pD[1].sum())   # (5,)

    # B[0] travelling-arrival column must stay in sync with D[0]:
    # when the agent arrives at a new patch its patch-type prior should equal D[0].
    # agent.B[0] has shape (1, 4, 4, 2) after the JAX batch broadcast.
    new_B0 = np.array(agent.B[0][0])           # (4, 4, 2) — drop batch dim to edit
    new_B0[:, 3, 0] = np.array(new_D0)
    new_B0[:, 3, 1] = np.array(new_D0)

    # Push all three updates back into the immutable agent in one tree_at call.
    agent = tree_at(
        lambda a: (a.D[0], a.D[1], a.B[0]),
        agent,
        (
            jnp.broadcast_to(new_D0[None], agent.D[0].shape),           # (1, 4)
            jnp.broadcast_to(new_D1[None], agent.D[1].shape),           # (1, 5)
            jnp.broadcast_to(jnp.array(new_B0)[None], agent.B[0].shape),# (1, 4, 4, 2)
        ),
    )

    #### LOGGING ####
    # Squeeze batch dim (0) and time dim (1) from final-step beliefs
    qs_final = [np.array(qs_history[t-1][f][0, -1]) for f in range(agent.num_factors)]

    qs_patch = qs_final[0][:3]                               # exclude travelling state (idx 3)
    qs_patch = qs_patch / qs_patch.sum()                     # renormalise

    log['episode'].append(episode)
    log['global_step'].append(step)
    log['patch_type'].append(world.current_patch.type)
    log['initial_resources'].append(world.current_patch.resources)
    log['residence_time'].append(t)
    log['food_obs_seq'].append([o[1] for o in obs_history])
    log['inferred_patch_type'].append(int(np.argmax(qs_patch)))
    log['qs_patch_type'].append(qs_patch)
    log['departure_obs'].append(obs_history[t-1])  # food obs that triggered leaving

    food_reward = {0: C[1][0], 1:  C[1][1], 2:  C[1][2], 3:  C[1][3]}  # mirrors C[1]
    travel_reward = np.concatenate(([0], np.repeat(-C[1][3], T))) # to offset "none" while travelling
    total_food = sum((food_reward[o[1]]+travel_reward[o[2]]) for o in obs_history[:t])
    log['reward_rate'].append(total_food / (t + T))

    # Log actual harvest and harvest rate
    log['harvest'].append(harvest_history)
    log['harvest_rate'].append(sum(harvest_history) / (t + T))

    # KL divergence: learned D[0] vs true patch-type prior (exclude travelling state)
    learned_D0 = np.array(agent.D[0][0])         # (4,) — drop batch dim
    kl_D0 = kl_div(true_D0[:3], learned_D0[:3])  # only the 3 foraging types
    log['kl_D0'].append(kl_D0)

    episode += 1
    step    += t + T

    if episode % 10 == 0:
        print(f"Episode {episode:4d} | Global step {step:6d}")

print("Main loop complete")

################################################################################################################################################################
# Analysis

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
from scipy.ndimage import uniform_filter1d

PATCH_NAMES  = {0: 'Rich', 1: 'Medium', 2: 'Sparse'}
PATCH_COLORS = {0: '#27ae60', 1: '#e67e22', 2: '#e74c3c'}

df = pd.DataFrame({
    'episode':            log['episode'],
    'global_step':        log['global_step'],
    'patch_type':         log['patch_type'],
    'initial_resources':  log['initial_resources'],
    'residence_time':     log['residence_time'],
    'inferred_type':      log['inferred_patch_type'],
})

################################################################################################################################################################
# MVT Computation

def compute_mvt(richness_dist, resource_dist, depletion_rate, travel_time,
                A1_matrix=None, C1_vector=None, C2_vector=None,
                n_sim: int = 30_000, max_t: int = 60, seed: int = SEED,
                tol: float = 1e-6, max_iter: int = 1000):
    rng = np.random.default_rng(seed)

    res_bins  = np.array([0., 15., 35., 65., 85., 100.])
    bin_sizes = np.array([15., 20., 30., 20., 15.])
    rl_thresholds = np.array([15., 35., 65., 85.])

    compute_util  = (A1_matrix is not None and C1_vector is not None)
    if compute_util:
        eu_per_rl = A1_matrix.T @ C1_vector
        # Sum C[2] over the T travel steps (indices 1..T); foraging is index 0 = 0.0
        travel_util = float(np.sum(C2_vector[1:travel_time + 1])) if C2_vector is not None else 0.0

    n_types     = len(richness_dist)
    gain_curves = np.zeros((n_types, max_t))
    util_curves = np.zeros((n_types, max_t)) if compute_util else None
    marginals   = np.zeros((n_types, max_t))

    for p in range(n_types):
        init_bins  = rng.choice(5, size=n_sim, p=resource_dist[:, p])
        res_idxs   = 4 - init_bins
        r0         = (res_bins[res_idxs]
                      + rng.uniform(size=n_sim) * bin_sizes[res_idxs])
        noise      = rng.beta(10, 10, size=(n_sim, max_t)) + 0.5
        r          = r0.copy()
        harvests   = np.zeros((n_sim, max_t))
        step_utils = np.zeros((n_sim, max_t)) if compute_util else None

        for t in range(max_t):
            dep            = np.clip(noise[:, t] * depletion_rate, 0., 1.)
            harvests[:, t] = r * dep
            r = np.maximum(0., r * (1. - dep))
            if compute_util:
                rl = 4 - np.clip(np.digitize(r, rl_thresholds), 0, 4)
                step_utils[:, t] = eu_per_rl[rl] 

        gain_curves[p] = np.cumsum(harvests,   axis=1).mean(axis=0)
        marginals[p]   = np.diff(gain_curves[p], prepend=0.0)
        if compute_util:
            util_curves[p] = np.cumsum(step_utils, axis=1).mean(axis=0) + travel_util

    times = np.arange(1, max_t + 1)

    def gain_at_inner(p, t_float):
        i    = int(t_float) - 1
        frac = t_float - int(t_float)
        if i + 1 < len(gain_curves[p]):
            return gain_curves[p][i] * (1 - frac) + gain_curves[p][i + 1] * frac
        return gain_curves[p][i]

    def best_t(R_bar_val):
        t_star = []
        for p in range(n_types):
            m          = marginals[p]
            candidates = np.where(m >= R_bar_val)[0]
            if len(candidates) == 0:
                t_star.append(1.0)
                continue
            i = candidates[-1]
            if i + 1 < len(m):
                frac = (m[i] - R_bar_val) / (m[i] - m[i + 1])
                t_star.append(float(i + 1) + frac)
            else:
                t_star.append(float(i + 1))
        return t_star

    t_init = [float(np.argmax(gain_curves[p] / (times + travel_time))) + 1.0
              for p in range(n_types)]
    R_bar  = (richness_dist @ np.array([gain_at_inner(p, t_init[p]) for p in range(n_types)])
              / (richness_dist @ np.array([t_init[p] + travel_time for p in range(n_types)])))

    for iteration in range(max_iter):
        t_star    = best_t(R_bar)
        R_bar_new = (richness_dist @ np.array([gain_at_inner(p, t_star[p]) for p in range(n_types)])
                     / (richness_dist @ np.array([t_star[p] + travel_time for p in range(n_types)])))
        if abs(R_bar_new - R_bar) < tol:
            R_bar = R_bar_new
            break
        R_bar = R_bar_new
    else:
        print(f"Warning: fixed-point iteration did not converge after {max_iter} steps")

    env_gain   = richness_dist @ gain_curves
    t_star_env = float(np.argmax(env_gain / (times + travel_time))) + 1.0

    return gain_curves, marginals, t_star, t_star_env, R_bar, times, util_curves


def gain_at(p, t_float, gain_curves):
    i    = int(t_float) - 1
    frac = t_float - int(t_float)
    if i + 1 < gain_curves.shape[1]:
        return gain_curves[p][i] * (1 - frac) + gain_curves[p][i + 1] * frac
    return gain_curves[p][i]


gain_curves, marginals, t_star, t_star_env, R_bar, mvt_times, util_curves = compute_mvt(
    richness_dist  = richness_dist,
    resource_dist  = resource_dist,
    depletion_rate = depletion_rate,
    travel_time    = T,
    A1_matrix      = A_true[1],
    C1_vector      = np.array(C[1]),
    C2_vector      = np.array(C[2]),
)

print(f"Converged R̄ = {R_bar:.4f}")
print("MVT Predictions (foraging steps until departure)")
print("─" * 44)
for p, name in PATCH_NAMES.items():
    agent_med = df[df['patch_type'] == p]['residence_time'].median()
    print(f"  {name:8s}  MVT t* = {t_star[p]:.2f}   agent median = {agent_med:.1f}")
print(f"  {'Env avg':8s}  MVT t* = {t_star_env:.2f}  (reference only)")

window = max(10, len(df) // 40)

################################################################################################################################################################
# Plots

# ──────────────────────────────────────────────────────────────────────────────
# Figure 1 — Theoretical MVT
#   Left:  classic gain-curve diagram with tangent line
#   Right: marginal gain vs shared R̄ threshold
# ──────────────────────────────────────────────────────────────────────────────
fig1, (ax_mvt, ax_marg) = plt.subplots(1, 2, figsize=(14, 5))
fig1.suptitle('Marginal Value Theorem — Theory',
              fontsize=13, fontweight='bold', y=1.02)

# Left: gain curves + tangent line
for p in range(3):
    color = PATCH_COLORS[p]
    g     = gain_curves[p]
    t_opt = t_star[p]

    ax_mvt.plot(mvt_times, g, color=color, lw=2.2, label=PATCH_NAMES[p])

    x_line = np.array([-T, t_opt + 2.0])
    y_line = R_bar * (x_line + T)
    ax_mvt.plot(x_line, y_line, '--', color=color, lw=1.4, alpha=0.7)

    ax_mvt.vlines(t_opt, 0, g[round(t_opt) - 1], colors=color, lw=1.0, ls=':', alpha=0.6)
    ax_mvt.scatter([t_opt], [g[round(t_opt) - 1]], color=color, zorder=5,
                   s=70, edgecolors='white', linewidths=0.8)
    ax_mvt.annotate(f"$t^*_{p}={t_opt:.1f}$", xy=(t_opt, g[round(t_opt) - 1]),
                    xytext=(t_opt + 0.6, g[round(t_opt) - 1] - 4),
                    fontsize=9, color=color)

ax_mvt.axvline(0, color='k', lw=0.9)
ax_mvt.axhline(0, color='k', lw=0.9)
ax_mvt.set_xlim(-T - 0.5, 35)
ax_mvt.set_ylim(-3, None)
ax_mvt.set_xlabel('Foraging steps $t$', fontsize=11)
ax_mvt.set_ylabel('$E[g(t)]$ – expected cumulative gain', fontsize=11)
ax_mvt.set_title(f'Gain curves & shared tangent\n$\\bar{{R}}={R_bar:.2f}$;  dot = $t^*$',
                 fontsize=11)
ax_mvt.legend(frameon=False, fontsize=10)

# Right: marginal gain vs R̄
for p in range(3):
    ax_marg.plot(mvt_times, marginals[p], color=PATCH_COLORS[p],
                 lw=1.8, label=PATCH_NAMES[p])
    ax_marg.axvline(t_star[p], color=PATCH_COLORS[p], lw=1.0, ls=':', alpha=0.7)

ax_marg.axhline(R_bar, color='k', lw=1.8, ls='--',
                label=f'$\\bar{{R}}={R_bar:.2f}$')
ax_marg.set_xlabel('Foraging steps $t$', fontsize=11)
ax_marg.set_ylabel('Marginal gain', fontsize=11)
ax_marg.set_title('Leave when marginal gain\ncrosses $\\bar{R}$ (shared threshold)', fontsize=11)
ax_marg.set_xlim(0, 30)
ax_marg.legend(frameon=False, fontsize=9)

plt.tight_layout()
plt.savefig('mvt_theory.png', bbox_inches='tight', dpi=150)
plt.show()
print("Saved → mvt_theory.png")

# ──────────────────────────────────────────────────────────────────────────────
# Figure 2 — Departure behaviour
#   Left:  residence-time IQR vs MVT predictions
#   Right: rolling-mean residence time per patch type (learning curve)
# ──────────────────────────────────────────────────────────────────────────────
fig2, (ax_dist, ax_learn) = plt.subplots(1, 2, figsize=(14, 5))
fig2.suptitle('Agent Departure Behaviour vs MVT Predictions',
              fontsize=13, fontweight='bold', y=1.02)

# Left: residence-time median + IQR whiskers
rng_jit = np.random.default_rng(1)
for p in range(3):
    rt     = df[df['patch_type'] == p]['residence_time'].values
    color  = PATCH_COLORS[p]
    jitter = rng_jit.uniform(-0.18, 0.18, len(rt))
    ax_dist.scatter(np.full(len(rt), p) + jitter, rt,
                    alpha=0.18, s=7, color=color, rasterized=True)

    q25, med, q75 = np.percentile(rt, [25, 50, 75])
    ax_dist.vlines(p, q25, q75, colors=color, lw=4.0, alpha=0.6)
    ax_dist.hlines(med, p - 0.25, p + 0.25, colors=color, lw=2.8,
                   label=f'{PATCH_NAMES[p]} median={med:.1f}')

    ax_dist.hlines(t_star[p], p - 0.35, p + 0.35, colors='k', lw=2.0, ls='--')
    ax_dist.annotate(f"MVT={t_star[p]:.2f}", xy=(p + 0.37, t_star[p]),
                     fontsize=8, va='center', color='#333')

ax_dist.set_xticks([0, 1, 2])
ax_dist.set_xticklabels([PATCH_NAMES[p] for p in range(3)], fontsize=10)
ax_dist.set_ylabel('Residence time (foraging steps)', fontsize=10)
ax_dist.set_title('Departure distribution by patch type\n'
                  '(bar = median, whiskers = IQR, -- = MVT)', fontsize=10)
ax_dist.legend(frameon=False, fontsize=9)

# Right: rolling-mean learning curve
for p in range(3):
    sub      = df[df['patch_type'] == p].reset_index(drop=True)
    rt       = sub['residence_time'].values.astype(float)
    smoothed = uniform_filter1d(rt, size=window, mode='nearest')
    ax_learn.plot(sub['episode'].values, smoothed,
                  color=PATCH_COLORS[p], lw=1.8, label=PATCH_NAMES[p])
    ax_learn.axhline(t_star[p], color=PATCH_COLORS[p], lw=1.0, ls='--', alpha=0.65)

ax_learn.set_xlabel('Episode', fontsize=10)
ax_learn.set_ylabel('Residence time (rolling mean)', fontsize=10)
ax_learn.set_title(f'Learning curve (window={window})\n(-- = MVT optimum per type)', fontsize=10)
ax_learn.legend(frameon=False, fontsize=9)

plt.tight_layout()
plt.savefig('departure_behaviour.png', bbox_inches='tight', dpi=150)
plt.show()
print("Saved → departure_behaviour.png")

# ──────────────────────────────────────────────────────────────────────────────
# Figure 3 — Inference & learning quality
#   Left:  patch-type posterior accuracy over episodes
#   Right: KL divergence of learned D[0] vs true patch-type distribution
# ──────────────────────────────────────────────────────────────────────────────
fig3, (ax_inf, ax_kl) = plt.subplots(1, 2, figsize=(14, 5))
fig3.suptitle('Inference Quality & Prior Learning',
              fontsize=13, fontweight='bold', y=1.02)

episodes = df['episode'].values
kl_window = window

# Left: patch-type inference accuracy
correct  = (df['inferred_type'] == df['patch_type']).astype(float)
smoothed = uniform_filter1d(correct.values, size=window, mode='nearest')
ax_inf.plot(episodes, smoothed, color='steelblue', lw=1.8, label='Accuracy (smoothed)')
ax_inf.fill_between(episodes, smoothed, alpha=0.15, color='steelblue')
ax_inf.axhline(1 / 3, color='grey', ls='--', lw=1.2, label='Chance (1/3)')
ax_inf.set_xlabel('Episode', fontsize=10)
ax_inf.set_ylabel('Fraction correct', fontsize=10)
ax_inf.set_title('Patch-type inference accuracy\n(final belief before leaving)', fontsize=10)
ax_inf.set_ylim(0, 1.05)
ax_inf.legend(frameon=False, fontsize=9)

# Right: KL divergence of learned D[0] vs true patch-type distribution
raw_kl   = np.array(log['kl_D0'])
smooth_kl = uniform_filter1d(raw_kl, size=kl_window, mode='nearest')

ax_kl.plot(episodes, raw_kl,   color='#8e44ad', alpha=0.20, lw=0.8)
ax_kl.plot(episodes, smooth_kl, color='#8e44ad', lw=2.2,
           label=f'Smoothed (w={kl_window})')
ax_kl.axhline(0, color='k', lw=1.0, ls='--', alpha=0.5, label='Convergence (KL=0)')
ax_kl.annotate(f'Start: {raw_kl[0]:.3f}',
               xy=(episodes[0], raw_kl[0]),
               xytext=(episodes[0] + len(episodes) * 0.03, raw_kl[0]),
               fontsize=8, color='#8e44ad')
ax_kl.annotate(f'End: {raw_kl[-1]:.3f}',
               xy=(episodes[-1], smooth_kl[-1]),
               xytext=(episodes[-1] - len(episodes) * 0.18,
                       smooth_kl[-1] + smooth_kl.max() * 0.05),
               fontsize=8, color='#8e44ad')
ax_kl.set_xlabel('Episode', fontsize=11)
ax_kl.set_ylabel('KL( $D^{true}_0$ || $D^{learned}_0$ )', fontsize=11)
ax_kl.set_title('Patch-type prior $D_0$ convergence\n(3 foraging states)', fontsize=11)
ax_kl.set_xlim(episodes[0], episodes[-1])
ax_kl.set_ylim(bottom=0)
ax_kl.legend(frameon=False, fontsize=9)

plt.tight_layout()
plt.savefig('inference_quality.png', bbox_inches='tight', dpi=150)
plt.show()
print("Saved → inference_quality.png")

# ──────────────────────────────────────────────────────────────────────────────
# Figure 4 — Agent performance vs MVT baselines
#   Left:  utility rate (C-weighted)
#   Right: raw harvest rate
# ──────────────────────────────────────────────────────────────────────────────

assert util_curves is not None, "util_curves is None — pass A1_matrix/C1_vector to compute_mvt"
 
fig4, (ax_util, ax_harv) = plt.subplots(1, 2, figsize=(14, 5))
fig4.suptitle('Agent Performance vs MVT Baselines', fontsize=13, fontweight='bold', y=1.02)
 
episodes_arr    = df['episode'].values
patch_types_arr = df['patch_type'].values
 
mvt_env_util   = np.array([gain_at(p, t_star_env, util_curves) / (t_star_env + T)
                            for p in patch_types_arr])
mvt_patch_util = np.array([gain_at(p, t_star[p],  util_curves) / (t_star[p]  + T)
                            for p in patch_types_arr])
mvt_env_harv   = np.array([gain_at(p, t_star_env, gain_curves) / (t_star_env + T)
                            for p in patch_types_arr])
mvt_patch_harv = np.array([gain_at(p, t_star[p],  gain_curves) / (t_star[p]  + T)
                            for p in patch_types_arr])
 
agent_util = np.array(log['reward_rate'])
agent_harv = np.array(log['harvest_rate'])
 
rate_window = max(20, len(df) // 30)
 
# Number of episodes used for the late-run summary scalar (last 33%)
n_late_rate = max(1, len(episodes_arr) // 3)
 
print("\nSummary scalars — mean over last 33% of episodes")
print("─" * 55)
 
for ax, a_vals, env_vals, patch_vals, ylabel, sub_a, sub_b, metric_name in [
    (ax_util,
     agent_util, mvt_env_util, mvt_patch_util,
     'Utility rate  [C(food) per episode step]',
     '(a)', '(b)', 'Utility rate'),
    (ax_harv,
     agent_harv, mvt_env_harv, mvt_patch_harv,
     'Harvest rate  [resource units per episode step]',
     '(c)', '(d)', 'Harvest rate'),
]:
    a_s = uniform_filter1d(a_vals,     size=rate_window, mode='nearest')
    e_s = uniform_filter1d(env_vals,   size=rate_window, mode='nearest')
    p_s = uniform_filter1d(patch_vals, size=rate_window, mode='nearest')
 
    ax.scatter(episodes_arr, a_vals, alpha=0.06, s=4, color='steelblue', rasterized=True)
    ax.plot(episodes_arr, a_s, color='steelblue', lw=2.2, label='Agent (actual)')
    ax.plot(episodes_arr, e_s, color='#222222', lw=1.8, ls='--',
            label=f'{sub_a} MVT env-level  '
                  f'($t^*_{{env}}={t_star_env:.1f}$, one threshold for all patches)')
    ax.plot(episodes_arr, p_s, color='darkorange', lw=1.8, ls=':',
            label=f'{sub_b} MVT per-patch  '
                  f'($t^*_p$: '
                  + ', '.join(f'{PATCH_NAMES[p]}={t_star[p]:.1f}' for p in range(3))
                  + ')')
    ax.fill_between(episodes_arr, a_s, p_s,
                    where=(p_s >= a_s), interpolate=True,
                    alpha=0.10, color='steelblue', label='Optimality gap')
 
    # Compute late-run summary means
    mean_agent = float(np.mean(a_vals[-n_late_rate:]))
    mean_env   = float(np.mean(env_vals[-n_late_rate:]))
    mean_patch = float(np.mean(patch_vals[-n_late_rate:]))
 
    # Relative deviation: (agent-MVT) / MVT (per-patch)
    reldev = (mean_agent - mean_patch) / np.abs(mean_patch) if mean_patch != 0 else float('nan')
 
    print(f"\n  {metric_name}")
    print(f"    Agent              : {mean_agent:.4f}")
    print(f"    MVT env-level      : {mean_env:.4f}")
    print(f"    MVT per-patch      : {mean_patch:.4f}")
    print(f"    Relative Deviation : {reldev*100:.1f}%")

    # Summary box
    summary_text = (
        f"Late-run means (33%)\n"
        f"  Agent         : {mean_agent:+.3f}\n"
        f"  MVT env-level : {mean_env:+.3f}\n"
        f"  MVT per-patch : {mean_patch:+.3f}\n"
        f"  Relative Deviation: {reldev*100:.1f}%"
    )
    ax.text(
        0.98, 0.04, summary_text,
        transform=ax.transAxes,
        ha='right', va='bottom',
        fontsize=8,
        family='monospace',
        bbox=dict(boxstyle='round,pad=0.45', facecolor='white',
                  alpha=0.88, edgecolor='#bbbbbb', lw=0.8)
    )
    # ─────────────────────────────────────────────────────────────────────────
 
    ax.set_xlabel('Episode', fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_xlim(episodes_arr[0], episodes_arr[-1])
    ax.legend(
        loc='lower left',
        frameon=False,
        fontsize=7
    )
 
ax_util.set_title(
    f'Utility rate over time  (window={rate_window})\n'
    '(a) vs env-level MVT   (b) vs per-patch MVT', fontsize=11)
ax_harv.set_title(
    f'Harvest rate over time  (window={rate_window})\n'
    '(c) vs env-level MVT   (d) vs per-patch MVT', fontsize=11)
 
plt.tight_layout()
plt.savefig('rate_comparison.png', bbox_inches='tight', dpi=150)
plt.show()
print("Saved → rate_comparison.png")