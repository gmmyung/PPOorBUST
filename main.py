from myppo import MyEnvironment, MyActor, MyCritic
from ppo import PPO
from typing import List
from matplotlib import lines, patches, animation, pyplot as plt
import numpy as np


def render(env: MyEnvironment, act: MyActor) -> None:
    fig, ax = plt.subplots(4, 8)
    assert isinstance(ax, np.ndarray)
    state = env.reset()
    action, _ = act.forward(state.unsqueeze(0))
    states, rewards = env.step(action.squeeze(0))
    actor_markers: List[patches.Circle] = []
    target_markers: List[patches.Circle] = []
    velocity_lines: List[lines.Line2D] = []
    reward_bars: List[patches.Rectangle] = []

    playback_frames = 1000

    def update(frame):
        action = act.forward(state.unsqueeze(0))[0].squeeze(0)
        states, rewards = env.step(action)
        for i in range(16):
            actor_markers[i].center = (states[i][0].item(), states[i][1].item())
            target_markers[i].center = (states[i][4].item(), states[i][5].item())

            velocity_lines[i].set_data(
                [states[i][0].item(), states[i][0].item() + states[i][2].item() * 10],
                [states[i][1].item(), states[i][1].item() + states[i][3].item() * 10],
            )

            reward_bars[i].set_height(rewards_subplot.get_ylim()[1] * rewards[i])

        if frame == playback_frames - 1:
            env.reset()

        return actor_markers + target_markers + velocity_lines

    for i in range(16):
        map_subplot = ax[i // 4][(i % 4) * 2]
        map_subplot.clear()
        actor_markers.append(
            patches.Circle(
                (states[i][0].item(), states[i][1].item()), 0.1, color="r", zorder=1
            )
        )
        target_markers.append(
            patches.Circle(
                (states[i][4].item(), states[i][5].item()), 0.1, color="g", zorder=2
            )
        )
        velocity_lines.append(
            map_subplot.plot(
                [states[i][0].item(), states[i][0].item() + states[i][2].item() * 10],
                [states[i][1].item(), states[i][1].item() + states[i][3].item() * 10],
                linewidth=1,
                color="b",
                zorder=0,
            )[0]
        )

        map_subplot.set_xlim(-1.5, 1.5)
        map_subplot.set_ylim(-1.5, 1.5)
        map_subplot.add_patch(actor_markers[i])
        map_subplot.add_patch(target_markers[i])
        map_subplot.set_aspect("equal")
        map_subplot.get_xaxis().set_visible(False)
        map_subplot.get_yaxis().set_visible(False)

        rewards_subplot = ax[i // 4][(i % 4) * 2 + 1]
        rewards_subplot.set_xlim(0, 0.2)
        rewards_subplot.set_ylim(0, 1)

        reward_bars.append(
            patches.Rectangle(
                (0, 0),
                rewards_subplot.get_xlim()[1],
                rewards_subplot.get_ylim()[1] * rewards[i],
            )
        )
        rewards_subplot.add_patch(reward_bars[i])
        rewards_subplot.set_aspect("equal")
        rewards_subplot.get_xaxis().set_visible(False)
        rewards_subplot.get_yaxis().set_visible(False)

    anim = animation.FuncAnimation(
        fig, update, frames=playback_frames, interval=1000 / 60
    )

    plt.show()


env = MyEnvironment(128, 1 / 30)
act = MyActor(env.get_action_dim(), env.get_state_dim(), env.get_actor_num())
crit = MyCritic(env.get_state_dim(), env.get_actor_num, 32, 4)
ppo = PPO(
    act,
    crit,
    env,
    clip_ratio=0.1,
    time_steps_per_epoch=128,
    discount_factor=0.99,
    actor_training_epochs=5,
    critic_training_epochs=5,
    actor_batch_size=32,
    critic_batch_size=32,
)

plt.ion()
mean_rewards: List[float] = []
time_steps: List[int] = []
fig, ax = plt.subplots()
assert isinstance(ax, plt.Axes)
for i in range(1000):
    mean_rewards.append(ppo.train())
    time_steps.append(i * env.get_actor_num() * 64)
    ax.clear()
    ax.plot(time_steps, mean_rewards)
    plt.show()
    plt.pause(0.001)

plt.ioff()
render(env, act)
