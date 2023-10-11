import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


# Plotting Utility functions
def plot_2d_image(img, cmap="viridis"):
    sns.heatmap(img, cmap=cmap, xticklabels=False, yticklabels=False)
    plt.show()


def plot_2d_images(
        recon_img, orig_img, show_rmse=True, rescale_for_rmse=True, title='', cmap="viridis", figsize=(12, 5)
):
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
        rmse = np.sqrt(np.mean((recon_img - orig_img) ** 2))
        plt.suptitle(f"{title}RMSE: {rmse:.4f}", y=1.02)

    plt.show()
