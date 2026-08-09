"""
CartPole REINFORCE with learned value baseline, implemented with PyTorch.

Task:
    Balance a pole on a cart. The agent observes the cart position,
    cart velocity, pole angle, and pole angular velocity, and must
    choose ONE of only two actions per step:
        action 0 -> push the cart to the LEFT
        action 1 -> push the cart to the RIGHT

Algorithm:
    REINFORCE (Monte Carlo policy gradient) with a learned, state-dependent
    value baseline:
        - a policy network outputs a probability distribution over actions
        - a value network V(s) estimates the expected discounted return from s
        - roll out a FULL episode, then compute the discounted returns G_t
        - advantage: A_t = G_t - V(s_t)   (V is detached, it is only a baseline)
        - policy loss:  -sum( log pi(a|s) * A_t )
        - value loss:   Huber(V(s_t), G_t)   (Monte Carlo regression)
    Unlike a constant per-episode baseline, a state-dependent baseline does
    not systematically punish late steps of a long episode: states that are
    inherently safe have high V(s), and near-failure states are judged by
    their own expected return.

Usage:
    python cartpole_pytorch.py           # train with a real-time plot
    python cartpole_pytorch.py --no-plot # train without the plot window
    python cartpole_pytorch.py --render  # render every training episode
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
LR_POLICY = 1e-2      # learning rate for the policy network
LR_VALUE = 1e-3       # learning rate for the value network (keep it gentle)
NUM_EPISODES = 1000   # total training episodes
EVAL_EPISODES = 5     # episodes used to report average return
PRINT_EVERY = 10      # print progress every N episodes
SOLVED_THRESHOLD = 450  # eval avg considered "solved"
SOLVED_STREAK = 3       # consecutive evaluations above the threshold
EARLY_STOP = True       # stop training as soon as it is solved
PLOT_FILENAME = "training_curve.png"


# ---------------------------------------------------------------------------
# Policy network: state -> action probabilities
# ---------------------------------------------------------------------------
class PolicyNet(nn.Module):
    """Fully-connected policy network.

    Input:  state vector (cart position, cart velocity, pole angle, pole
            angular velocity) -> 4 numbers
    Output: probability of each action -> 2 numbers (LEFT, RIGHT)
            produced by a softmax over the action logits
    """

    def __init__(self, state_dim: int = 4, action_dim: int = 2):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # Softmax turns the raw logits into a valid probability distribution.
        return torch.softmax(self.fc3(x), dim=-1)


# ---------------------------------------------------------------------------
# Value network: state -> expected discounted return V(s)
# ---------------------------------------------------------------------------
class ValueNet(nn.Module):
    """Fully-connected value network used as the advantage baseline.

    Input:  4-dimensional state
    Output: single scalar V(s), the expected discounted return from state s.
    No activation on the output because V(s) can be any real number.
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
# REINFORCE agent with a learned value baseline
# ---------------------------------------------------------------------------
class REINFORCEAgent:
    """Policy-gradient agent. Learns the policy and a value baseline."""

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
        Returns (action, log_prob). log_prob is None in greedy mode.
        """
        state_tensor = torch.tensor(
            state, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        probs = self.policy_net(state_tensor)

        if greedy:
            return int(probs.argmax(dim=1).item()), None

        # Categorical distribution over {LEFT, RIGHT} with the network's probs.
        dist = distributions.Categorical(probs)
        action = dist.sample()
        return int(action.item()), dist.log_prob(action)

    def update(self, states, log_probs, returns):
        """One REINFORCE update using the learned V(s) as baseline.

        Policy loss:  -sum( log pi(a|s) * (G_t - V(s_t)) )
        Value loss:   Huber(V(s_t), G_t)

        The baseline is detached from the policy loss, so the policy
        gradient remains unbiased (REINFORCE with baseline) while variance
        is reduced by the state-dependent value estimate.
        """
        states_tensor = torch.tensor(
            np.array(states), dtype=torch.float32, device=self.device
        )
        returns_tensor = torch.tensor(
            returns, dtype=torch.float32, device=self.device
        )

        # ---- Value update: regress V(s_t) toward the Monte Carlo return ----
        values = self.value_net(states_tensor).squeeze(1)
        value_loss = F.smooth_l1_loss(values, returns_tensor)
        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()

        # ---- Policy update: advantage A_t = G_t - V(s_t) ----
        # values.detach() means the baseline only scales the gradient,
        # it does not get improved by the policy objective itself.
        advantages = returns_tensor - values.detach()
        policy_loss = -torch.cat(log_probs).mul(advantages).sum()
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        return policy_loss.item(), value_loss.item()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def compute_discounted_returns(rewards, gamma: float = GAMMA):
    """Convert step rewards into discounted returns G_t.

    G_t = r_t + gamma * r_{t+1} + gamma^2 * r_{t+2} + ...
    Computed backwards in one pass for efficiency.
    """
    returns = []
    discounted_return = 0.0
    for reward in reversed(rewards):
        discounted_return = reward + gamma * discounted_return
        returns.insert(0, discounted_return)
    return returns


def evaluate(agent: REINFORCEAgent, env: gym.Env, episodes: int = EVAL_EPISODES) -> float:
    """Run a few greedy episodes to measure current policy performance."""
    total = 0.0
    for _ in range(episodes):
        state, _ = env.reset(seed=random.randint(0, 1_000_000))
        episode_reward = 0.0
        done = False
        while not done:
            action, _ = agent.select_action(state, greedy=True)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward
        total += episode_reward
    return total / episodes


def train(agent: REINFORCEAgent, render: bool, plot: bool = True):
    env = gym.make("CartPole-v1", render_mode="human" if render else None)
    # A fresh instance is used for evaluation so the two are independent.
    eval_env = gym.make("CartPole-v1")
    plotter = TrainingPlotter(enabled=plot)

    best_average = 0.0
    solved_streak = 0
    for episode in range(NUM_EPISODES):
        states = []     # every state visited in this episode
        log_probs = []  # log pi(a_t | s_t) for every step of this episode
        rewards = []    # immediate rewards for every step of this episode

        state, _ = env.reset(seed=random.randint(0, 1_000_000))
        done = False

        # ---- Episode rollout: collect a complete trajectory ----
        while not done:
            action, log_prob = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            states.append(state)
            log_probs.append(log_prob)
            rewards.append(reward)
            state = next_state

        # ---- REINFORCE update after the episode ends ----
        returns = compute_discounted_returns(rewards)
        policy_loss, value_loss = agent.update(states, log_probs, returns)

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
        description="CartPole REINFORCE with learned value baseline (PyTorch)"
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

    agent = REINFORCEAgent(device)
    train(agent, render=args.render, plot=args.plot)


if __name__ == "__main__":
    main()
