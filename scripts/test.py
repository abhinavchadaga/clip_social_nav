from dataset import CLIPDataModule
import torch
from torch import nn
import numpy as np

# sample joystick values
j_1_1 = (1.6, 0, 0)
j_1_2 = (1.6, 0, -0.5)
j_1_3 = (1.6, 0, 0.25)

s1 = np.array([j_1_1, j_1_2, j_1_3], dtype=np.float32)
print(f"shape: {s1.shape}")

j_2_1 = (0.74, 0, 0.17)
j_2_2 = (1.6, 0, 0)
j_2_3 = (1.6, 0, 0)

s2 = np.array([j_2_1, j_2_2, j_2_3], dtype=np.float32)


# de-emphasize messages that represent 'go straight with max velocity'
# forward_velocity weight = 2.0 - abs(j[0]). Higher the forward velocity, lower the weight
# angular velocity weight = abs(j[2]) * some arbitrary weight

# for an entire sample
# find mean and std_deviation of abs(forward velocity)
# emphasize lower means, and higher std_deviations for forward velocity

# find mean and std deviation of abs(angular velocity)
# emphasize higher mean and higher std_deviations

# collect goal information for associated joystick sample
# emphasize lower x goal, higher abs(y goal)


def generate_weight(joystick_sample: np.ndarray, goal_sample: np.ndarray,
                    fwd_weight: float, ang_weight: float) -> float:
    joystick_sample = np.abs(joystick_sample)
    lin_x_mu, ang_z_mu = np.mean(joystick_sample, axis=0)
    lin_x_sig, ang_z_sig = np.std(joystick_sample, axis=0)
    mu_weight = (2.0 - lin_x_mu) * fwd_weight + ang_z_mu * ang_weight
    sig_weight = lin_x_sig + ang_z_sig
    goal_x, goal_y = goal_sample
    goal_x_weight, goal_y_weight = 12.0 - goal_x, abs(goal_y)
    return mu_weight + sig_weight + goal_x_weight + goal_y_weight


# define batch size
batch_size = 10

# load and setup data
dm = CLIPDataModule(
    data_path="/home/abhinavchadaga/CS/clip_social_nav/data/",
    batch_size=batch_size,
    num_workers=2,
    verbose=True,
)

dm.setup()

# load a batch
batch = next(iter(dm.train_dataloader()))
joystick = batch[1]
goals = batch[2]

weights = []
for i in range(batch_size):
    joystick_sample = joystick[i, :, :].detach().cpu().numpy()
    goal_sample = goals[i, :].detach().cpu().numpy()
    w = generate_weight(joystick_sample, goal_sample, 1.0, 1.2)
    weights.append(w)

print(weights)
