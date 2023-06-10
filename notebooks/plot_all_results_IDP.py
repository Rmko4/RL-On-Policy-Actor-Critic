# %%
from matplotlib import style
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import re

# %%
results_dir = Path("../results")
all_result_path = results_dir/"Inverted-Double-Pendulum_all.csv"

pattern = r"train\d_(.*)_.*"

# %%
df = pd.read_csv(all_result_path)


# %%
# Extract the columns that end with 'rewards
col_names = [col for col in df.columns if col.endswith('rewards')]

# Extract the alg_name from the column names
alg_names = [re.match(pattern, col).group(1) for col in col_names]


# %%
df = df[df['frame_count'] <= 1000000]

# Create a new DataFrame with the extracted columns
grouped_data = df[col_names].groupby(
    alg_names, axis=1).apply(lambda x: x.values)

frames = df["frame_count"].values



# %%
style.use(["cleanplot", "font_libertine"])


# %%
fig_width = 3.26
fig_height = 3

# Set the window size for the moving average
window_size = 50

# Create the figure object with the custom size
fig = plt.figure(figsize=(fig_width, fig_height))

for alg_name in grouped_data.keys():
    data = grouped_data[alg_name]

    # Get the indices where there are NaN values
    nan_indices = np.isnan(data)
    nan_indices_remove = np.all(nan_indices, axis=1)

    # Remove the NaN values from the array
    data = data[~nan_indices_remove]

    nan_indices = np.isnan(data)

    for i in range(1, len(data)):
        if np.any(nan_indices[i]):
            for j in range(len(data[i])):
                if nan_indices[i][j]:
                    data[i][j] = data[i-1][j]

    frames_wo_nan = frames[~nan_indices_remove]


    mu = np.mean(data, axis=1)
    sigma = np.std(data, axis=1)

    padded_mu = np.pad(mu, (window_size-1, 0), mode='edge')
    padded_sigma = np.pad(sigma, (window_size-1, 0), mode='edge')
    ma = np.convolve(padded_mu, np.ones(window_size)/window_size, mode='valid')
    ma_sd = np.convolve(padded_sigma, np.ones(
        window_size)/window_size, mode='valid')

    plt.plot(frames_wo_nan, ma, label=alg_name, lw=1)
    plt.fill_between(frames_wo_nan, ma + 0.5*ma_sd, ma - 0.5*ma_sd, alpha=0.3)

plt.xlabel("Test Epoch")
plt.ylabel("Average Reward")
plt.legend()
fig.savefig(results_dir/"IDP_results_all.pdf", dpi=300, bbox_inches='tight')
plt.show()


# %%
