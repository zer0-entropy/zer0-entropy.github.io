---
title: 'One-Page Summary: Reinforcement Learning Foundations & PPO'
date: 2026-06-29
permalink: /posts/2026/06/rl-foundations-ppo/
tags:
  - reinforcement-learning
  - ppo
  - robotics
---

### 1. Actor vs. Critic: The Step-by-Step Execution Loop

The Actor-Critic architecture splits the reinforcement learning problem into two specialized modules that collaborate in a real-time execution pipeline:

* **Step 1: Look and Act (The Actor, $\pi_	heta(a|s)$):** The agent observes the current state $s_t$. The Actor network processes this input and outputs an action strategy. The agent executes action $a_t$ in the environment.
* **Step 2: Experience the Outcome:** The environment processes the action, shifting the agent into a **new state** ($s_{t+1}$) and handing back an **immediate reward** ($r_t$).
* **Step 3: Guess the Future (The Critic, $V_\phi(s)$):** The Critic network evaluates the new situation. It looks at the fresh state $s_{t+1}$ and predicts how many total rewards can be collected from that point onward ($V_\phi(s_{t+1})$).
* **Step 4: Calculate the Reality Check (The Advantage, $A_t$):** We combine hard reality with our future guess to evaluate the past state: $	ext{Target Value} = r_t + \gamma V_\phi(s_{t+1})$. The Critic subtracts its original prediction of the past state ($V_\phi(s_t)$) from this target to calculate the **Advantage**:

$$A_t = \left( r_t + \gamma V_\phi(s_{t+1}) ight) - V_\phi(s_t)$$

* **Step 5: Update Both Networks:** * *The Critic* updates its weights via Mean Squared Error loss to bring its old guess ($V_\phi(s_t)$) closer to the reality-backed target anchor.
  * *The Actor* uses the Advantage signal ($A_t$) to scale its gradients—increasing the probability of the action if $A_t > 0$, or suppressing it if $A_t < 0$.

---

### 2. Why PPO is Stable: The Clipped Objective

Traditional policy gradient updates modify the Actor's weights using raw log probabilities scaled by the Advantage ($\mathcal{L} = -\log \pi_	heta \cdot A_t$). If an action yields a massive positive advantage, a single massive gradient step can completely warp the policy parameters—leading to a **tragic policy collapse** where the agent enters an irreversible performance death spiral.

PPO (Proximal Policy Optimization) secures the **Actor's Update Step (Step 5)** with a mathematical safety harness:

$$\mathcal{L}_{	ext{CLIP}}(	heta) = \min\left( r_t(	heta)A_t, \;	ext{clip}(r_t(	heta), 1-\epsilon, 1+\epsilon)A_t ight)$$

Where $r_t(	heta) = rac{\pi_	heta(a_t|s_t)}{\pi_{	heta_{	ext{old}}}(a_t|s_t)}$ is the probability ratio between the updating policy and the old policy that collected the data.

* **The Guardrail:** If an action was amazing ($A_t > 0$), the network tries to aggressively increase $r_t(	heta)$. However, the moment $r_t(	heta)$ exceeds $1+\epsilon$ (typically $1.2$), the `clip` function activates and locks it.
* **The Result:** Taking the minimum ($\min$) ensures that the update stops rewarding the network for moving too far from its safe baseline. By keeping changes strictly **incremental and localized**, PPO safely passes over the same data buffer multiple times without letting the policy jump off a mathematical cliff.

---

### 3. Why PPO is Widely Used in Robotics

PPO is the industry-standard baseline for complex physics simulations and real-world robotic locomotion (such as training quadrupeds) because of how it handles this specific loop:

* **High Sample Efficiency from the Data Loop:** Because the clipping mechanism locks changes within a safe boundary, PPO can reuse the exact same trajectory buffer for multiple training epochs (e.g., 10 to 80 mini-batch passes). This extracts maximum learning progression out of computationally heavy simulation environments (like MuJoCo or Isaac Sim) before discarding the data.
* **Resilience to Contact Noise during Steps 2 & 3:** Robotic locomotion involves high-dimensional, continuous joint control spaces with sudden contact dynamics (foot slips, uneven terrain). The conservative updates ensure that when a noisy or jarring transition occurs in **Step 2**, the resulting extreme Advantage values cannot instantly break the policy.
* **Algorithmic Simplicity in Network Updates:** Unlike older stable methods (like TRPO) that require calculating heavy, unstable second-order mathematical derivatives to constrain policy steps, PPO achieves its rock-solid stability using a standard, first-order gradient descent loss. It is incredibly lightweight, easy to scale across massive parallel GPU simulation architectures, and highly forgiving to hyperparameter variations.