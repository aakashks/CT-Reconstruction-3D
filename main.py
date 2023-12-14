from utils import *
from core import *

def generate_projections(
    rots, vol_recon, file_path="./data/generated/", clip=(10, 17), factor=100
):
    """
    Generate projections for a given list of rotations
    
    Parameters
    ----------
    rots
        list of rotations to be generated
    vol_recon
        reconstructed volume
    file_path
        path to stored intercept matrices
    clip
        clip the images to this range
    factor
        scale the images by the factor (default 100 used in core.py)
    """
    imgs = []
    for rot in rots:
        A = torch.load(os.path.join(file_path, f"matrix_rot_{rot}.pt"))
        proj = torch.sparse.mm(A, vol_recon.flatten().view(-1, 1))
        img = proj.view(200, 200) * factor
        img = torch.clip(img, min=clip[0], max=clip[1])
        imgs.append(img)

    return imgs


def main(rots, file_path='./data/'):
    
    # we are solving the forward problem here
    # for Ax = b we are testing the A matrix for provided reconstruction volume's projection ie. x
    # here we are dealing with each rotation separately
    
    # load data from pt files
    projections = torch.load(file_path + "projections_scaled.pt")
    recons = torch.load(file_path + "recon_scaled.pt")

    # the original reconstructions were reduced from 200x200x200 to a smaller number in z dimension
    # due to this the reconstructed volume is padded with zeros to make it 200x200x200
    recons_filled = torch.zeros(200, 200, 200)
    recons_filled[12:191, :, :] = recons

    # generate list of projections for the given rotations
    generated_projections = generate_projections(
        rots, vol_recon=recons_filled, file_path="./data/generated/matrix_200/"
    )
    
    # plot all the generated projections in a single figure to get a rough idea of correctness
    plot_images_line(generated_projections)
    plt.savefig(
        "./results/multiple_rotations_generated.png", dpi=600, bbox_inches="tight"
    )

    # plot the generated projections against the original projections
    for i in range(len(rots)):
        plot_2d_comparison(
            generated_projections[i],
            projections[:, :, rots[i]],
            title=f"Rotation {rots[i]}   ",
        )
        plt.savefig(
            f"./results/generated_vs_projections_comparison/rotation_{rots[i]}.png",
            dpi=600,
            bbox_inches="tight",
        )


if __name__ == "__main__":
    main(rots=[0, 1, 10, 20, 30, 40, 50])
