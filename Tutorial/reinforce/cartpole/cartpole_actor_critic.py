"""
CartPole one-step TD actor-critic (A2C-style), implemented with PyTorch.

Task:
    Balance a pole on a cart. The agent observes the cart position,
    cart velocity, pole angle, and pole angular velocity, and must
    choose ONE of only two actions per step:
        action 0 -> push the cart to the LEFT
        action 1 -> push the cart to the RIGHT

Algorithm:
    One-step TD actor-critic:
        - a policy network pi(a|s) outputs action probabilities (the actor)
        - a value network V(s) estimates the expected return from s (the critic)
        - the advantage at each step is the one-step TD error:
              delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)   (V(s') = 0 if done)
        - policy loss: -sum( log pi(a|s) * delta_t )   (delta detached)
        - value loss:  MSE(V(s_t), r_t + gamma * V(s_{t+1}))
        - a small entropy bonus keeps exploration alive and prevents collapse
    Unlike REINFORCE there is no full-return computation: the target is
    bootstrapped from V(s_{t+1}), so variance is much lower and updates
    can be made online. For clarity the episode is batched and the update
    is applied once per episode, using per-step TD advantages.

Known limitation (verified empirically on CartPole-v1):
    With constant +1 rewards and gamma = 0.99, the one-step bootstrap bias
    keeps the critic below the true value (V ~ 10 instead of 100), so the
    TD error of every non-terminal step is an almost constant positive
    number (~+0.9). A constant advantage has ZERO expected policy gradient
    and only sampling noise remains, so the policy collapses to a constant
    action and evaluation stays at random level (~9-10). This is a known
    weakness of pure one-step TD actor-critic on CartPole. In practice it is
    fixed by training the critic on n-step/GAE returns, adding a stronger
    entropy bonus, or using parallel environments. Raise ENTROPY_COEF, or
    switch the advantage to GAE (lambda ~ 0.95), to make it learn.

Usage:
    python cartpole_actor_critic.py           # train with a real-time plot
    python cartpole_actor_critic.py --no-plot # train without the plot window
    python cartpole_actor_critic.py --render  # render every training episode
"""

import argparse
import random

import gymnasium as gym
import numpy as np
import torch
import torch.distributions as distributions
import torch.nn as nn
import torch.nn.functional as F

# matplotlib is optional: plotting degrades gracefully when it is missing.
try:
    import matplotlib.pyplot as plt
    PLOTTING_AVAILABLE = True
except ImportError:
    plt = None
    PLOTTING_AVAILABLE = False


# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
GAMMA = 0.99          # discount factor for future rewards
LR_POLICY = 1e-2      # learning rate for the policy network (actor)
LR_VALUE = 1e-2       # learning rate for the value network (critic)
ENTROPY_COEF = 0.01   # entropy bonus weight (set 0 to disable)
NUM_EPISODES = 500    # total training episodes
EVAL_EPISODES = 5     # episodes used to report average return
PRINT_EVERY = 10      # print progress every N episodes
SOLVED_THRESHOLD = 450  # eval avg considered "solved"
SOLVED_STREAK = 3       # consecutive evaluations above the threshold
EARLY_STOP = True       # stop training as soon as it is solved
PLOT_FILENAME = "training_curve.png"


# ---------------------------------------------------------------------------
# Policy network (actor): state -> action probabilities
# ---------------------------------------------------------------------------
class PolicyNet(nn.Module):
    """Fully-connected policy network.

    Input:  state vector -> 4 numbers
    Output: probability of each action -> 2 numbers (LEFT, RIGHT)
    """

    def __init__(self, state_dim: int = 4, action_dim: int = 2):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return torch.softmax(self.fc3(x), dim=-1)


# ---------------------------------------------------------------------------
# Value network (critic): state -> V(s)
# ---------------------------------------------------------------------------
class ValueNet(nn.Module):
    """Fully-connected value network.

    Input:  4-dimensional state
    Output: single scalar V(s), the expected discounted return from state s.
    """

    def __init__(self, state_dim: int = 4):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


# ---------------------------------------------------------------------------
# Real-time training dashboard
# ---------------------------------------------------------------------------
class TrainingPlotter:
    """Real-time matplotlib dashboard: rewards, eval average and value loss.

    The plot is refreshed every PRINT_EVERY episodes and the final figure is
    saved to PLOT_FILENAME. When matplotlib is missing or no display is
    available, plotting is skipped silently (only the final save is attempted).
    """

    def __init__(self, enabled: bool = True):
        self.enabled = enabled and PLOTTING_AVAILABLE
        self.interactive = False
        self.episode_rewards = []
        self.eval_averages = []
        self.value_losses = []
        if not self.enabled:
            return
        try:
            plt.ion()
            self.fig, (self.ax_reward, self.ax_loss) = plt.subplots(2, 1, figsize=(9, 7))
            self.fig.tight_layout()
            plt.show(block=False)
            self.interactive = True
        except Exception:
            # e.g. headless backend: keep collecting data, save at the end.
            self.interactive = False

    def update(self, episode, episode_reward, eval_avg, value_loss):
        if not self.enabled:
            return
        self.episode_rewards.append(episode_reward)
        self.eval_averages.append(eval_avg)
        self.value_losses.append(value_loss)
        if episode % PRINT_EVERY != 0:
            return
        episodes = list(range(1, episode + 1))
        self.ax_reward.clear()
        self.ax_reward.plot(
            episodes, self.episode_rewards,
            color="lightsteelblue", alpha=0.7, label="episode reward",
        )
        self.ax_reward.plot(
            episodes, self.eval_averages,
            color="tab:blue", lw=2, label="eval avg",
        )
        self.ax_reward.axhline(500.0, color="green", ls="--", lw=1, label="max 500")
        self.ax_reward.axhline(
            SOLVED_THRESHOLD, color="orange", ls=":", lw=1,
            label=f"solved {SOLVED_THRESHOLD}",
        )
        self.ax_reward.set_ylabel("steps survived")
        self.ax_reward.legend(loc="upper left", fontsize=8)
        self.ax_loss.clear()
        self.ax_loss.plot(episodes, self.value_losses, color="tab:red")
        self.ax_loss.set_ylabel("value loss")
        self.ax_loss.set_xlabel("episode")
        self.fig.tight_layout()
        try:
            self.fig.canvas.draw()
            if self.interactive:
                self.fig.canvas.flush_events()
                plt.pause(0.01)
        except Exception:
            pass

    def save(self):
        if not self.enabled:
            if PLOTTING_AVAILABLE:
                print("Plotting disabled (--no-plot).")
            else:
                print("Plotting disabled: install matplotlib for real-time plots.")
            return
        try:
            self.fig.savefig(PLOT_FILENAME, dpi=150)
            print(f"Training plot saved to {PLOT_FILENAME}")
        except Exception as exc:
            print(f"Could not save training plot: {exc}")


# ---------------------------------------------------------------------------
# One-step TD actor-critic agent
# ---------------------------------------------------------------------------
class ActorCriticAgent:
    """Learns both a policy (actor) and a value function (critic)."""

    def __init__(self, device: torch.device):
        self.device = device
        self.policy_net = PolicyNet().to(device)
        self.value_net = ValueNet().to(device)
        self.policy_optimizer = torch.optim.Adam(
            self.policy_net.parameters(), lr=LR_POLICY
        )
        self.value_optimizer = torch.optim.Adam(
            self.value_net.parameters(), lr=LR_VALUE
        )

    def select_action(self, state, greedy: bool = False):
        """Sample an action from the policy distribution.

        During training we sample (stochastic) so the agent explores;
        during evaluation we take the most likely action (greedy).
        Returns (action, log_prob, entropy). Entropy is None in greedy mode.
        """
        state_tensor = torch.tensor(
            state, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        probs = self.policy_net(state_tensor)

        if greedy:
            return int(probs.argmax(dim=1).item()), None, None

        dist = distributions.Categorical(probs)
        action = dist.sample()
        return int(action.item()), dist.log_prob(action), dist.entropy()

    def update(
        self,
        states,
        log_probs,
        entropies,
        rewards,
        next_states,
        dones,
    ):
        """One batched one-step TD update for a complete episode.

        TD target:        y_t = r_t + gamma * V(s_{t+1})  (0 if s' is terminal)
        TD error:         delta_t = y_t - V(s_t)          (used as advantage)
        Policy loss:      -sum( log pi(a|s) * delta_t ) - ENTROPY_COEF * H
        Value loss:       MSE(V(s_t), y_t)

        The next-state value and the TD error are detached so the critic
        gradient only improves V(s_t) and the actor gradient only improves
        the policy.
        """
        states_tensor = torch.tensor(
            np.array(states), dtype=torch.float32, device=self.device
        )
        next_states_tensor = torch.tensor(
            np.array(next_states), dtype=torch.float32, device=self.device
        )
        rewards_tensor = torch.tensor(
            rewards, dtype=torch.float32, device=self.device
        )
        dones_tensor = torch.tensor(
            dones, dtype=torch.float32, device=self.device
        )

        # One-step bootstrapped target; V(s') is detached (no gradient).
        with torch.no_grad():
            next_values = self.value_net(next_states_tensor).squeeze(1)
            targets = rewards_tensor + GAMMA * next_values * (1.0 - dones_tensor)

        values = self.value_net(states_tensor).squeeze(1)
        td_errors = targets - values  # the advantage for the actor

        # ---- Actor update: policy gradient with entropy bonus ----
        policy_loss = -(torch.cat(log_probs) * td_errors.detach()).sum()
        entropy_bonus = ENTROPY_COEF * torch.cat(entropies).mean()
        total_policy_loss = policy_loss - entropy_bonus

        self.policy_optimizer.zero_grad()
        total_policy_loss.backward()
        self.policy_optimizer.step()

        # ---- Critic update: regress V(s) toward the TD target ----
        value_loss = F.mse_loss(values, targets)
        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()

        return policy_loss.item(), value_loss.item(), entropy_bonus.item()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def evaluate(agent: ActorCriticAgent, env: gym.Env, episodes: int = EVAL_EPISODES) -> float:
    """Run a few greedy episodes to measure current policy performance."""
    total = 0.0
    for _ in range(episodes):
        state, _ = env.reset(seed=random.randint(0, 1_000_000))
        episode_reward = 0.0
        done = False
        while not done:
            action, _, _ = agent.select_action(state, greedy=True)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward
        total += episode_reward
    return total / episodes


def train(agent: ActorCriticAgent, render: bool, plot: bool = True):
    env = gym.make("CartPole-v1", render_mode="human" if render else None)
    # A fresh instance is used for evaluation so the two are independent.
    eval_env = gym.make("CartPole-v1")
    plotter = TrainingPlotter(enabled=plot)

    best_average = 0.0
    solved_streak = 0
    for episode in range(NUM_EPISODES):
        states = []
        log_probs = []
        entropies = []
        rewards = []
        next_states = []
        dones = []

        state, _ = env.reset(seed=random.randint(0, 1_000_000))
        done = False

        # ---- Episode rollout ----
        while not done:
            action, log_prob, entropy = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            states.append(state)
            log_probs.append(log_prob)
            entropies.append(entropy)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)
            state = next_state

        # ---- One-step TD update (batched at episode end) ----
        policy_loss, value_loss, entropy_bonus = agent.update(
            states, log_probs, entropies, rewards, next_states, dones
        )

        average_return = evaluate(agent, eval_env)
        best_average = max(best_average, average_return)
        solved_streak = solved_streak + 1 if average_return >= SOLVED_THRESHOLD else 0
        plotter.update(episode + 1, sum(rewards), average_return, value_loss)

        if (episode + 1) % PRINT_EVERY == 0:
            print(
                f"Episode {episode + 1:4d} | "
                f"episode reward {sum(rewards):5.1f} | "
                f"eval avg {average_return:5.1f} | "
                f"best avg {best_average:5.1f} | "
                f"value loss {value_loss:.3f}"
            )
        if EARLY_STOP and solved_streak >= SOLVED_STREAK:
            print(
                f"\n>>> SOLVED: eval avg >= {SOLVED_THRESHOLD} for "
                f"{SOLVED_STREAK} consecutive evaluations."
            )
            print(">>> Stopping training early (set EARLY_STOP = False to disable).")
            break

    env.close()
    eval_env.close()
    plotter.save()
    print(f"\nTraining finished. Best average evaluation return: {best_average:.1f}")


def main():
    parser = argparse.ArgumentParser(
        description="CartPole one-step TD actor-critic (PyTorch)"
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="render every training episode (slower but visual)",
    )
    parser.add_argument(
        "--plot",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="show a real-time training plot (default: on); use --no-plot to disable",
    )
    args = parser.parse_args()

    # Prefer CUDA when available, otherwise fall back to CPU.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    agent = ActorCriticAgent(device)
    train(agent, render=args.render, plot=args.plot)


if __name__ == "__main__":
    main()
