"""Regenerate the article figures: python generate_figures.py.

Requires numpy, matplotlib, pillow, and (for the last figure) torch.
Includes a complete PyTorch training example using the loss shown in the post.
"""
from pathlib import Path
import os
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "flow-matching-matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "images"
OUT.mkdir(exist_ok=True)
BLUE, ORANGE, INK, GRAY = "#386cb0", "#d07836", "#263342", "#8c96a0"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11,
    "text.color": INK, "axes.labelcolor": INK, "axes.edgecolor": "#d0d5da",
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.color": GRAY, "ytick.color": GRAY, "svg.fonttype": "none",
    "savefig.facecolor": "white", "figure.facecolor": "white",
})


def save(fig, name):
    fig.savefig(OUT / name, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)


# The opening animation is inspired by the density-contour toy example on
# slide 16 of MIT's 2026 Lecture 2 (figure credited there to Yaron Lipman).
# It is an original reconstruction, not an extraction of the slide's GIF.
# Transport N(0, I) to two tilted Gaussian lobes on the right. All particles
# follow the exact marginal field for independent linear noise/data pairing.
SIGMA = 0.35
FLOW_MEANS = np.array([[3.7, 1.15], [4.15, -1.05]])


def covariance(angle, major, minor):
    c, s = np.cos(angle), np.sin(angle)
    rotation = np.array([[c, -s], [s, c]])
    return rotation @ np.diag([major ** 2, minor ** 2]) @ rotation.T


FLOW_COVARIANCES = np.array([
    covariance(np.deg2rad(118), 0.66, 0.25),
    covariance(np.deg2rad(38), 0.64, 0.36),
])


def flow_components(t, x):
    covariances = (1 - t) ** 2 * np.eye(2) + t ** 2 * FLOW_COVARIANCES
    inverse = np.linalg.inv(covariances)
    offsets = x[:, None, :] - t * FLOW_MEANS[None, :, :]
    log_density = (
        -0.5 * np.einsum("nki,kij,nkj->nk", offsets, inverse, offsets)
        -0.5 * np.linalg.slogdet(covariances)[1][None, :]
        -np.log(2 * np.pi) - np.log(2)
    )
    return inverse, offsets, log_density


def flow_density(t, x):
    return np.exp(flow_components(t, x)[2]).sum(axis=1)


def velocity(t, x):
    inverse, offsets, log_density = flow_components(t, x)
    weights = np.exp(log_density - log_density.max(axis=1, keepdims=True))
    weights /= weights.sum(axis=1, keepdims=True)
    rate = (t * FLOW_COVARIANCES - (1 - t) * np.eye(2)) @ inverse
    conditional = FLOW_MEANS[None, :, :] + np.einsum("kij,nkj->nki", rate, offsets)
    return np.sum(weights[..., None] * conditional, axis=1)


rng = np.random.default_rng(7)
x = rng.normal(size=(260, 2))
frames = [x.copy()]
dt = 1 / 320
for k in range(320):
    t = k * dt
    k1 = velocity(t, x)
    k2 = velocity(t + dt / 2, x + dt * k1 / 2)
    k3 = velocity(t + dt / 2, x + dt * k2 / 2)
    k4 = velocity(t + dt, x + dt * k3)
    x = x + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6
    if (k + 1) % 4 == 0:
        frames.append(x.copy())

frames = np.array(frames)
highlight = np.argmin(np.linalg.norm(frames[0] - [-0.5, -0.55], axis=1))
fig, ax = plt.subplots(figsize=(8.6, 5.6))
fig.subplots_adjust(left=0.055, right=0.945, bottom=0.18, top=0.88)
ax.set(xlim=(-3.4, 6.5), ylim=(-3.3, 3.3), xlabel="x", ylabel="y")
ax.set_aspect("equal")
gx, gy = np.meshgrid(np.linspace(-3.4, 6.5, 200), np.linspace(-3.3, 3.3, 135))
grid = np.stack([gx.ravel(), gy.ravel()], axis=-1)
levels = [0.006, 0.016, 0.035, 0.065, 0.105, 0.16, 0.23, 0.32, 0.44, 0.60]
for t in [0, 1]:
    ax.contour(gx, gy, flow_density(t, grid).reshape(gx.shape),
               levels=[0.025, 0.075, 0.13], colors=GRAY,
               linestyles="dashed", linewidths=0.8, alpha=0.5, zorder=1)
ax.text(0, 2.85, r"$p_{\mathrm{init}}$", ha="center", fontsize=19, color=INK)
ax.text(4.05, 2.85, r"$p_{\mathrm{data}}$", ha="center", fontsize=19, color=INK)
contours = []
scatter = ax.scatter(*frames[0].T, s=9, c=INK, alpha=0.55, edgecolors="none", zorder=4)
trail, = ax.plot([], [], color=BLUE, lw=2, zorder=5)
point = ax.scatter(*frames[0, highlight], s=70, c=BLUE, edgecolors="white", linewidths=1.2, zorder=6)
ax.scatter(*frames[0, highlight], s=65, facecolors="white", edgecolors=BLUE, linewidths=1.5, zorder=5)
ax.annotate(r"$X_0$", xy=frames[0, highlight], xytext=(-1.5, -2.1), fontsize=16,
            color=BLUE, ha="center", arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.2))
moving_label = ax.annotate("", xy=frames[0, highlight], xytext=(9, 12),
                           textcoords="offset points", fontsize=16, color=BLUE, zorder=7,
                           bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))
ax.set_title("A flow moves samples from noise to data", loc="left", fontsize=13, pad=20)
clock = ax.set_title("t = 0.00", loc="right", fontsize=12, pad=20)
fig.text(0.5, 0.045, "Orange contours: current density    ·    Blue trail: one sample's path",
         ha="center", fontsize=10, color=INK)


def update(i):
    for contour in contours:
        contour.remove()
    contours.clear()
    density = flow_density(i / 80, grid).reshape(gx.shape)
    contours.append(ax.contourf(gx, gy, density, levels=levels, cmap="Oranges", alpha=0.8, zorder=2))
    contours.append(ax.contour(gx, gy, density, levels=levels[:-1], colors="#aa622d", linewidths=0.7, alpha=0.85, zorder=3))
    scatter.set_offsets(frames[i])
    trail.set_data(frames[:i + 1, highlight, 0], frames[:i + 1, highlight, 1])
    point.set_offsets(frames[i, highlight][None, :])
    moving_label.xy = frames[i, highlight]
    moving_label.set_text("" if i == 0 else (r"$X_1$" if i == 80 else r"$X_t$"))
    clock.set_text(f"t = {i / 80:.2f}")
    return scatter, trail, point, moving_label, clock


animation = FuncAnimation(fig, update, frames=[0] * 16 + list(range(81)) + [80] * 28, interval=80)
animation.save(OUT / "flow.gif", writer=PillowWriter(fps=16), dpi=120)
plt.close(fig)

# Euler approximation against an exact rotating, contracting trajectory.
def exact(t):
    return np.exp(-0.7 * t)[:, None] * np.stack([np.cos(2.4 * t), np.sin(2.4 * t)], -1)


fig, axes = plt.subplots(1, 2, figsize=(9, 4.3), sharex=True, sharey=True, layout="constrained")
curve = exact(np.linspace(0, 1, 400))
for ax, steps in zip(axes, [4, 20]):
    ax.plot(*curve.T, c=GRAY, lw=2, label="Exact trajectory")
    points = [np.array([1.0, 0.0])]
    for _ in range(steps):
        p = points[-1]
        v = np.array([-0.7 * p[0] - 2.4 * p[1], 2.4 * p[0] - 0.7 * p[1]])
        points.append(p + v / steps)
    points = np.array(points)
    ax.plot(*points.T, "o-", c=BLUE, ms=4, lw=1.3, label="Euler approximation")
    for start, end in zip(points[:-1], points[1:]):
        ax.annotate("", xy=end, xytext=start, arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.2))
    ax.scatter([1], [0], c=INK, s=35, zorder=5)
    ax.set(title=f"{steps} Euler steps", xlabel="x", aspect="equal", xlim=(-1.2, 1.15), ylim=(-0.1, 1.15))
    ax.grid(alpha=0.15)
axes[0].set_ylabel("y")
axes[1].legend(loc="lower left", fontsize=9)
save(fig, "euler-simulation.svg")

# These one-dimensional densities use the same target mixture as the code.
fig, axes = plt.subplots(1, 4, figsize=(10, 3.2), sharex=True, sharey=True, layout="constrained")
xgrid = np.linspace(-4, 4, 600)
for ax, t in zip(axes, [0, 0.35, 0.7, 1]):
    variance = (1 - t) ** 2 + (SIGMA * t) ** 2
    parts = [0.5 * np.exp(-(xgrid - t * m) ** 2 / (2 * variance)) / np.sqrt(2 * np.pi * variance) for m in [-2, 2]]
    for part, color in zip(parts, [BLUE, ORANGE]):
        ax.plot(xgrid, part, "--", c=color, lw=1.4)
    ax.plot(xgrid, sum(parts), c=INK, lw=2)
    ax.fill_between(xgrid, sum(parts), color=INK, alpha=0.05)
    ax.set(title=f"t = {t:g}", xlabel="x", xlim=(-4, 4), ylim=(0, 0.65), xticks=[-3, 0, 3])
axes[0].set_ylabel("Probability density")
save(fig, "probability-path.svg")

# Local flow illustration inspired by page 20 of MIT's 2026 Lecture 2.
# https://diffusion.csail.mit.edu/2026/docs/20260122_Lecture_02.pdf#page=20
# Curves and arrows use the same smooth field; no flux magnitudes are implied.
fig, ax = plt.subplots(figsize=(6.8, 5.2), layout="constrained")
grid = np.linspace(-2.2, 2.2, 160)
xx, yy = np.meshgrid(grid, grid)
vx, vy = 0.65 * np.sin(yy), 1 + 0.15 * yy
ax.streamplot(grid, grid, vx, vy, color="#dfb28d", density=0.85,
              linewidth=1, arrowsize=0.001, zorder=1)
qx, qy = np.meshgrid(np.linspace(-2, 2, 11), np.linspace(-2, 2, 11))
ax.quiver(qx, qy, 0.65 * np.sin(qy), 1 + 0.15 * qy,
          color="#b5bbc1", angles="xy", scale_units="xy", scale=4.5,
          width=0.0035, headwidth=3.8, zorder=2)

cx, cy, half_width, half_height = -0.55, 0.0, 0.45, 0.35
ax.add_patch(plt.Rectangle((cx - half_width, cy - half_height),
                          2 * half_width, 2 * half_height,
                          facecolor="white", edgecolor=INK, lw=1.8, zorder=3))
ax.scatter([cx], [cy], s=18, color=INK, zorder=5)
ax.text(cx + 0.13, cy - 0.08, r"$\mathbf{x}$", fontsize=24, zorder=5)

for boundary_y, color, label, label_y in [
    (-half_height, BLUE, "Inflow", -0.95),
    (half_height, ORANGE, "Outflow", 0.95),
]:
    # Highlight motion across the lower and upper boundary, tangent to u.
    slope = 0.65 * np.sin(boundary_y) / (1 + 0.15 * boundary_y)
    start_dy, end_dy = (-0.65, 0.10) if boundary_y < 0 else (-0.10, 0.65)
    for boundary_x in [cx - 0.3, cx, cx + 0.3]:
        start = (boundary_x + slope * start_dy, boundary_y + start_dy)
        end = (boundary_x + slope * end_dy, boundary_y + end_dy)
        ax.annotate("", xy=end, xytext=start,
                    arrowprops=dict(arrowstyle="-|>", color=color,
                                    lw=2.6, mutation_scale=18), zorder=4)
    ax.text(0.15, label_y, label, color=color, fontsize=17, weight="bold",
            ha="left", va="center", zorder=5,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.9, pad=3))

ax.text(1.35, 1.75, r"$u_t(x)$", fontsize=22, zorder=5,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.9, pad=3))
ax.set(xlim=(-2.2, 2.2), ylim=(-2.2, 2.2), aspect="equal",
       xlabel="x", ylabel="y", xticks=[], yticks=[])
ax.set_title("Flow through a small region around x", fontsize=15, pad=12)
save(fig, "continuity-and-marginalization.svg")

# Exact example: different trajectories, identical one-time distributions.
def trajectory_scale(t):
    return np.sqrt((1 - t) ** 2 + t ** 2 / 4)


def trajectory_marginal_field(t, x):
    return 2 + (5 * t / 4 - 1) / trajectory_scale(t) ** 2 * (x - 2 * t)


times = np.linspace(0, 1, 301)
starting_point, endpoint = -1.0, 3.0
conditional_trajectory = (1 - times) * starting_point + times * endpoint
marginal_trajectory = 2 * times + trajectory_scale(times) * starting_point

# Verify that integrating the displayed field produces the analytical flow.
check_initial = np.array([-2.0, -1.0, 0.0, 1.5])
check_x = check_initial.copy()
check_dt = 1 / 400
for k in range(400):
    t = k * check_dt
    k1 = trajectory_marginal_field(t, check_x)
    k2 = trajectory_marginal_field(t + check_dt / 2, check_x + check_dt * k1 / 2)
    k3 = trajectory_marginal_field(t + check_dt / 2, check_x + check_dt * k2 / 2)
    k4 = trajectory_marginal_field(t + check_dt, check_x + check_dt * k3)
    check_x += check_dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6
    next_t = (k + 1) * check_dt
    assert np.allclose(check_x, 2 * next_t + trajectory_scale(next_t) * check_initial,
                       atol=1e-8, rtol=0)

fig, axes = plt.subplots(1, 2, figsize=(9, 3.9))
fig.subplots_adjust(left=0.08, right=0.98, bottom=0.25, top=0.81, wspace=0.3)
fig.suptitle("Different trajectories, same distribution", fontsize=16, y=0.98)
left, right = axes
left.plot(times, conditional_trajectory, color=BLUE, lw=2.2,
          label="Conditional interpolation")
left.plot(times, marginal_trajectory, color=ORANGE, lw=2.2, linestyle="--",
          label="Marginal ODE")
left.scatter([0], [starting_point], color=INK, s=25, zorder=4)
left.text(0.04, -0.94, r"$x_0=-1$", fontsize=10)
left.scatter([1, 1], [endpoint, marginal_trajectory[-1]],
             color=[BLUE, ORANGE], s=25, zorder=4)
left.axvline(0.5, color=GRAY, alpha=0.45, lw=1, linestyle=":")
left.set(title="One sample, starting from the same point",
         xlabel="Time t", ylabel="Position x", xlim=(-0.02, 1.04), ylim=(-1.3, 3.3))

snapshot_t = 0.5
positions = np.linspace(-1.1, 3.1, 500)
variance = (1 - snapshot_t) ** 2 + snapshot_t ** 2 / 4
conditional_density = np.exp(-(positions - 2 * snapshot_t) ** 2 / (2 * variance)) / np.sqrt(2 * np.pi * variance)
# Change of variables for Y_t = 2t + s_t X_0 with standard Gaussian X_0.
scale = trajectory_scale(snapshot_t)
initial_positions = (positions - 2 * snapshot_t) / scale
marginal_density = np.exp(-initial_positions ** 2 / 2) / (np.sqrt(2 * np.pi) * scale)
assert np.allclose(conditional_density, marginal_density, atol=1e-14, rtol=0)
right.plot(positions, conditional_density, color=BLUE, lw=2.8)
right.plot(positions, marginal_density, color=ORANGE, lw=2.2, linestyle="--")
right.set(title="All samples: density at t = 0.5", xlabel="Position x",
          ylabel="Probability density", xlim=(-1.1, 3.1), ylim=(0, 0.8))
right.text(1, 0.15, "The two densities overlap", ha="center", fontsize=10)
fig.legend(*left.get_legend_handles_labels(), loc="lower center", ncol=2,
           frameon=False, bbox_to_anchor=(0.5, 0.015), fontsize=11)
save(fig, "marginalization-trajectories.svg")

# Complete toy training example using the loss shown in the post.
import torch
from torch import nn

torch.set_num_threads(2)
torch.manual_seed(7)


class VectorField(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 64),  # Two coordinates plus time.
            nn.SiLU(),
            nn.Linear(64, 64),
            nn.SiLU(),
            nn.Linear(64, 2),
        )

    def forward(self, x, t):
        return self.net(torch.cat([x, t], dim=-1))


def sample_data(batch_size):
    centers = torch.tensor([[-2.0, 0.0], [2.0, 0.0]])
    labels = torch.randint(0, 2, (batch_size,))
    return centers[labels] + 0.35 * torch.randn(batch_size, 2)


model = VectorField()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
model.train()
for step in range(3000):
    z = sample_data(256)
    x0 = torch.randn_like(z)
    t = torch.rand(z.shape[0], 1)
    xt = (1 - t) * x0 + t * z
    target_velocity = z - x0
    predicted_velocity = model(xt, t)
    loss = (predicted_velocity - target_velocity).square().sum(-1).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


@torch.no_grad()
def sample(model, num_samples=2000, num_steps=100):
    model.eval()
    x = torch.randn(num_samples, 2)
    dt = 1.0 / num_steps
    for k in range(num_steps):
        t = torch.full((num_samples, 1), k * dt)
        x = x + dt * model(x, t)
    return x


generated = sample(model).numpy()
data = sample_data(2000).numpy()
assert generated.shape == (2000, 2) and np.isfinite(generated).all()
fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), sharex=True, sharey=True, layout="constrained")
for ax, values, label in zip(axes, [data, generated], ["Target data", "Generated · 100 Euler steps"]):
    ax.scatter(*values.T, s=3, c=BLUE, alpha=0.25, edgecolors="none", rasterized=True)
    ax.set(title=label, xlabel="x", xlim=(-3.6, 3.6), ylim=(-1.7, 1.7), aspect="equal")
axes[0].set_ylabel("y")
save(fig, "pytorch-samples.svg")
for sign in [-1, 1]:
    cluster = generated[generated[:, 0] * sign > 0]
    print(f"Cluster {sign:+}: mass={len(cluster)/len(generated):.3f}, mean={cluster.mean(0)}, std={cluster.std(0)}")
print("Figures saved to", OUT)
