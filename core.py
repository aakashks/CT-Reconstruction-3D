import gc
import logging
import os

import torch
from tqdm.auto import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class CreateInterceptMatrix:
    def __init__(self, detector_plate_length, source_to_object, source_to_detector, pixel_size, projections,
                 dtype=torch.float32, resolution=None):
        """
        Parameters
        ----------
        detector_plate_length
            no of detectors fit in 1 side length of square detector plate
        source_to_object
            source to the centre of object
        source_to_detector
            source to any detector's centre (all detectors should be equidistance from source)
        pixel_size
            size of 1 pixel unit in the grid of pixels (of both object and detector, which is an ASSUMPTION)
        projections
            total no of projections taken (rotations taken)
        resolution
            ASSUMING same resolution along all axis
        dtype
            float16 gives bad precision, distorted images
            prefer using float32 only
        """
        self.dl = detector_plate_length
        # deliberatly scaling my space by factor of 1/100 so that precision in float32 is better
        self.sod = source_to_object / 100
        self.sdd = source_to_detector / 100
        self.ps = pixel_size / 100
        self.p = projections

        # Assumption: maximum angle is 2pi
        self.phi = 2 * torch.pi / projections
        self.n = resolution if resolution is not None else detector_plate_length
        if self.n != self.dl:
            logging.warning('different detector length and resolution arent supported yet, so might cause errors')

        assert self.n % 2 == 0, 'if needed revert changes from previous commit'
        self.dtype = dtype
        self.cache_ic_rot0_list = None

    def intercept_coords(self, detector_coords, dtype=torch.float32, eps=1e-10):
        """
        get all voxel intercept_coords for 1 ray or a batch of rays
        batches should be on one ray param only for eg. beta only
        line parameters are taken using centre of object as origin

        Parameters
        ----------
        detector_coords
            must be torch tensor on cpu

        Returns
        -------
        hybrid sparse coo tensor object. size (batches, n*n*n, 3). where last dim is dense
        where for each batch / row X, Y, Z are flat indexed as they are expected to be
        """

        alpha, betas = (detector_coords[0].view(-1, 1).to(device, dtype),
                        detector_coords[1].view(-1, 1).to(device, dtype))

        # each pixel is represented by its bottom left corner coordinate
        ps = self.ps
        n = self.n
        lambd = self.sod
        gamma = self.sdd

        # creating the grid of voxels
        a = torch.arange((-n // 2) * ps, (n // 2) * ps, ps, device=device, dtype=dtype)
        X, Y, Z = torch.meshgrid(a, a, a, indexing='xy')

        # flatten
        X = X.flatten().view(1, -1)
        Y = Y.flatten().view(1, -1)
        Z = Z.flatten().view(1, -1)

        # Defining equations which will generate rays acc to the parametric form
        def xy_from_z(z):
            return (
                -alpha*(z-lambd)/(gamma + eps),
                -betas*(z-lambd)/(gamma+eps)
            )

        def yz_from_x(x):
            return (
                -betas*(x)/(-alpha+eps),
                lambd + gamma*(x)/(-alpha+eps)
            )

        def zx_from_y(y):
            return (
                lambd - gamma*y/(betas + eps),
                -alpha*y/(betas + eps)
            )

        # find intercept coordinates at planes
        def intercepts_z_plane(x0, y0, z):
            x, y = xy_from_z(z)
            # only True value locations are stored in mask (sparse)
            mask = ((x0 <= x) & (x < x0 + ps) & (y0 <= y) & (y < y0 + ps)).to_sparse()
            del x0, y0
            i = torch.stack([(mask * x).values(), (mask * y).values(), (mask * z).values()], -1)
            del x, y, z
            return torch.sparse_coo_tensor(mask.indices(), i, (*(mask.size()), 3), dtype=i.dtype, device=device)

        def intercepts_x_plane(x, y0, z0):
            y, z = yz_from_x(x)
            mask = ((y0 <= y) & (y < y0 + ps) & (z0 <= z) & (z < z0 + ps)).to_sparse()
            del y0, z0
            i = torch.stack([(mask * x).values(), (mask * y).values(), (mask * z).values()], -1)
            del x, y, z
            return torch.sparse_coo_tensor(mask.indices(), i, (*(mask.size()), 3), dtype=i.dtype, device=device)

        def intercepts_y_plane(x0, y, z0):
            z, x = zx_from_y(y)
            mask = ((x0 <= x) & (x < x0 + ps) & (z0 <= z) & (z < z0 + ps)).to_sparse()
            del x0, z0
            i = torch.stack([(mask * x).values(), (mask * y).values(), (mask * z).values()], -1)
            del x, y, z
            return torch.sparse_coo_tensor(mask.indices(), i, (*(mask.size()), 3), dtype=i.dtype, device=device)

        # get intercept_coords with 6 boundaries of each voxel
        I = [intercepts_z_plane(X, Y, Z), intercepts_z_plane(X, Y, Z + ps),
             intercepts_y_plane(X, Y, Z), intercepts_y_plane(X, Y + ps, Z),
             intercepts_x_plane(X, Y, Z), intercepts_x_plane(X + ps, Y, Z)]

        del X, Y, Z
        del betas
        del alpha
        gc.collect()
        torch.cuda.empty_cache()
        return I

    def detector_coordinates(self, dtype=torch.float32):
        """
        generate rays from source to each detector
        returns shape (2, alphas, betas)
        where alphas = betas = dl = n
        """
        n = self.dl
        ps = self.ps
        x = torch.arange((-n + 1) * ps, n * ps, 2 * ps, device=device, dtype=dtype) / 2
        detector_coords = torch.stack(torch.meshgrid(x, x, indexing='xy'), 0)
        del x
        gc.collect()
        torch.cuda.empty_cache()
        return detector_coords.to('cpu')

    def intercept_coords_rot_0(self, k, save=True, file_path='', dtype=torch.float32):
        """
        for only rotation 0
        cache in RAM a hybrid sparse tensor of shape (alphas * betas, x * y * z, 3)
        x = y = z = n
        with approx n elements for each (alphas, betas, :)

        k
            no of betas together
            a suitable value can be 100 for colab and kaggle (16GBs). 50 for 8GB (on n=200 case)
            meshgrid size causes the limitation on k
        """
        assert self.n % k == 0
        all_rays_rot0 = self.detector_coordinates(dtype=dtype)
        list_sparse_matrix = [torch.empty([0, self.n**3, 3], dtype=dtype, device=device).to_sparse(2) for _ in range(6)]

        for alpha_i in tqdm(range(self.dl), leave=False):
            for betas_i in range(self.dl // k):
                # peak GPU RAM usage
                list_k_rows = self.intercept_coords(all_rays_rot0[:, alpha_i, betas_i * k:(betas_i + 1) * k])
                # concatenating increments size at that dimension and automatically adjusts the indices
                for i in range(6):
                    list_sparse_matrix = torch.cat([list_sparse_matrix[0], list_k_rows[0]], dim=0)
                del list_k_rows

        # store the intercept coords for further calc
        self.cache_ic_rot0_list = list_sparse_matrix
        del list_sparse_matrix

        # save for future reference
        if save:
            torch.save(self.cache_ic_rot0_list, os.path.join(file_path, 'intercept_coords.pt'))

        gc.collect()
        torch.cuda.empty_cache()

    def intercept_matrix_rots(self, rots, file_path='', dtype=torch.float32):
        """
        stores few rows of the intercept matrix generated by rotating intercept_coords of rotation 0
        indexing is assumed as follows
        for each row (x, y, z).flatten()
        and rows are (phis, alphas, betas).flatten()
        """
        n = self.n
        for rotation in tqdm(rots, desc='Generating matrix'):
            phi = torch.tensor(rotation * self.phi)
            c, s = torch.cos(phi), torch.sin(phi)
            # for a rotation
            rotation_matrix = torch.tensor([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=dtype, device=device).T
            I = [self.cache_ic_rot0_list[i].coalesce().to(device) for i in range(6)]

            values = [intercept_coords_0.values().mm(rotation_matrix) for intercept_coords_0 in I]

            I_r = [torch.sparse_coo_tensor(I[i].indices(), values[i], size=(self.dl*self.dl, n*n*n)) for i in range(6)]
            intercept_coords_0 = torch.abs(torch.abs(torch.abs(I_r[0] - I_r[1]) - torch.abs(I_r[2] - I_r[3])) - torch.abs(I_r[4] - I_r[5]))

            intercept_lengths = torch.sparse_coo_tensor(
                intercept_coords_0.indices(), torch.norm(intercept_coords_0.values(), dim=1),
                size=(self.dl*self.dl, n*n*n))

            torch.save(intercept_lengths.cpu(), os.path.join(file_path, f'matrix_rot_{rotation}.pt'))

            del intercept_coords_0, intercept_lengths, rotation_matrix
            # clean up memory
            gc.collect()
            torch.cuda.empty_cache()


# functions
def generate_sinogram(rots, vol_recon, file_path, clip=(10, 17), factor=100):
    imgs = []
    for rot in rots:
        A = torch.load(os.path.join(file_path, f'matrix_rot_{rot}.pt')).to(device)
        proj = torch.sparse.mm(A, vol_recon.to(device).flatten().view(-1, 1))
        img = proj.view(200, 200) * factor
        img = torch.clip(img, min=clip[0], max=clip[1])
        imgs.append(img.cpu())

    return imgs
