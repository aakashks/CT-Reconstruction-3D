import math

import matplotlib.pyplot as plt
import torch
import time
import gc
import seaborn as sns
from typing import List

# Measure GPU performance
# measure performance of any function
def measure_performance(func, repeats=10, *args, **kwargs):
    time_taken = []
    for _ in range(repeats):
        t0 = time.time()
        func(*args, **kwargs)
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()
        execution_time = time.time() - t0
        time_taken.append(execution_time)

    time_taken = torch.tensor(time_taken)
    print(f'--- Time metrics for {func.__name__} ---')
    print(f'Mean   = {time_taken.mean().item():.3f}s')
    print(f'Median = {time_taken.median().item():.3f}s')
    print(f'Max    = {time_taken.max().item():.3f}s')


# Plotting Utility functions
def plot_2d_images(images: List[torch.Tensor], ncols=3, cbar=True, figsize=None, cmap="viridis"):
    n = len(images)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 3*nrows) if figsize is None else figsize)
    axes = axes.reshape(nrows, ncols)
    for i in range(n):
        sns.heatmap(images[i], ax=axes[i // ncols, i % ncols], square=True, cbar=cbar, cmap=cmap, xticklabels=False, yticklabels=False)

    plt.tight_layout()
    plt.show()


def plot_images_line(images):
    fig, axs = plt.subplots(ncols=len(images), squeeze=False)
    for i, img in enumerate(images):
        axs[0, i].imshow(img)
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])


def plot_2d_comparison(
        recon_img, orig_img, show_rmse=True, rescale_for_rmse=True, title='', cmap="viridis", figsize=(12, 5)
):
    """
    compare generated image with a given known image
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    for ax, img in zip(axes, [recon_img, orig_img]):
        sns.heatmap(img, cmap=cmap, ax=ax, xticklabels=False, yticklabels=False)

    fig.tight_layout()
    axes[0].set_title("Reconstructed")
    axes[1].set_title("Original")

    if show_rmse:
        if rescale_for_rmse:
            # Rescale the images
            [recon_img, orig_img] = [
                (x - x.min()) / (x.max() - x.min()) for x in [recon_img, orig_img]
            ]
        rmse = torch.sqrt(torch.mean((recon_img - orig_img) ** 2))
        plt.suptitle(f"{title}RMSE: {rmse:.4f}", y=1.02)

    plt.show()
