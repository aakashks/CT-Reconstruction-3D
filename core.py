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
        # deliberately scaling my space by factor of 1/100 so that precision in float32 is better
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

    @staticmethod
    def write_iv_to_storage(sparse_matrix, rot_no, file_path='./data/generated/'):
        # write to 2 files indices and values
        torch.save(sparse_matrix, os.path.join(file_path, f'matrix_rot_{rot_no}.pt'))
        del sparse_matrix

    def intercepts_for_rays(self, ray_coords, dtype=None, eps=1e-10):
        """
        get all voxel intercepts for 1 ray or a batch of rays
        batches should be on one ray param only for eg. beta only
        line parameters are taken using centre of object as origin

        Parameters
        ----------
        ray_coords
            must be torch tensor on cpu

        Returns
        -------
        sparse_coo Tensor object. size (batches, n*n*n).
        where for each batch / row X, Y, Z are flat indexed as they are expected to be
        """

        phi, alpha, betas = (ray_coords[0].view(-1, 1),
                             ray_coords[1].view(-1, 1),
                             ray_coords[2].view(-1, 1))

        # option to change data type within method
        dtype = dtype if dtype is not None else self.dtype

        alpha = alpha.to(device, dtype)
        betas = betas.to(device, dtype)
        phi = phi.to(device, dtype)

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

        sin = torch.sin(phi)
        cos = torch.cos(phi)

        # slopes in parametric form of the 3d ray
        mx = gamma*sin - alpha*cos
        mz = gamma*cos + alpha*sin

        xi = lambd*sin
        zi = lambd*cos

        # Defining equations which will generate rays acc to the parametric form
        def xy_from_z(z):
            return (
                xi + mx*(z-zi)/(mz + eps),
                -betas*(z-zi)/(mz+eps)
            )

        def yz_from_x(x):
            return (
                -betas*(x-xi)/(mx+eps),
                zi + mz*(x-xi)/(mx+eps)
            )

        def zx_from_y(y):
            return (
                zi - mz*y/(betas + eps),
                xi - mx*y/(betas + eps)
            )

        # find intercept coordinates at planes
        def intercepts_z_plane(x0, y0, z):
            x, y = xy_from_z(z)
            # only True value locations are stored in mask (sparse)
            mask = ((x0 <= x) & (x < x0 + ps) & (y0 <= y) & (y < y0 + ps)).to_sparse_coo()
            del x0, y0
            i = torch.stack([mask * x, mask * y, mask * z], 0)
            del mask, x, y, z
            return i

        def intercepts_x_plane(x, y0, z0):
            y, z = yz_from_x(x)
            mask = ((y0 <= y) & (y < y0 + ps) & (z0 <= z) & (z < z0 + ps)).to_sparse_coo()
            del y0, z0
            i = torch.stack([mask * x, mask * y, mask * z], 0)
            del mask, x, y, z
            return i

        def intercepts_y_plane(x0, y, z0):
            z, x = zx_from_y(y)
            mask = ((x0 <= x) & (x < x0 + ps) & (z0 <= z) & (z < z0 + ps)).to_sparse_coo()
            del x0, z0
            i = torch.stack([mask * x, mask * y, mask * z], 0)
            del mask, x, y, z
            return i

        # get intercepts with 6 boundaries of each voxel
        C1 = torch.abs(intercepts_z_plane(X, Y, Z) - intercepts_z_plane(X, Y, Z + ps))
        C2 = torch.abs(intercepts_y_plane(X, Y, Z) - intercepts_y_plane(X, Y + ps, Z))
        C3 = torch.abs(intercepts_x_plane(X, Y, Z) - intercepts_x_plane(X + ps, Y, Z))

        del betas
        del X, Y, Z
        del alpha, phi, mx, mz, xi, zi

        # To get length of line from all these intercept coordinates
        # first we take |x2 - x1|
        ic = torch.abs(torch.abs(C1 - C2) - C3)
        del C1, C2, C3
        # now squaring will give the length
        intercept_lengths = torch.sqrt(ic[0] ** 2 + ic[1] ** 2 + ic[2] ** 2)
        del ic

        # move from GPU
        il = intercept_lengths.to('cpu')
        del intercept_lengths
        gc.collect()
        torch.cuda.empty_cache()
        return il

    def generate_rays(self, phis, dtype=torch.float32):
        """
        generate rays from source to each detector for a rotation angle of phi
        returns shape (3, phis, alphas, betas)
        where alphas = betas = dl = n
        """
        n = self.dl
        ps = self.ps
        x = torch.arange((-n + 1) * ps, n * ps, 2 * ps, device=device, dtype=dtype) / 2
        detector_coords = torch.stack(torch.meshgrid(x, x, indexing='xy'), 0)

        phis = phis.to(device, dtype).view(-1, 1, 1)

        alphas = detector_coords[0] + torch.zeros_like(phis)
        betas = detector_coords[1] + torch.zeros_like(phis)
        phis = phis + torch.zeros_like(alphas)

        line_params_tensor = torch.stack([phis, alphas, betas], 0)

        del phis, alphas, betas, detector_coords, x
        gc.collect()
        torch.cuda.empty_cache()
        return line_params_tensor.to('cpu')

    def intercept_matrix_per(self, rotation, k, all_rays_rot, dtype=torch.float32):
        """
        for only 1 rotation
        write to storage sparse tensor of shape (alphas, betas, x, y, z)
        x = y = z = n
        with approx n elements for each (alphas, betas, :)

        all_rays_rot
            all rays for 1 rotation. shape (3, alphas, betas)

        k
            no of betas together
            a suitable value can be 100 for colab and kaggle (16GBs). 50 for 8GB (on n=200 case)

        """
        assert self.n % k == 0

        sparse_matrix = torch.sparse_coo_tensor(size=[0, self.n**3], dtype=dtype)

        for alpha_i in tqdm(range(self.dl), leave=False):
            for betas_i in range(self.dl // k):
                # peak GPU RAM usage
                k_rows = self.intercepts_for_rays(all_rays_rot[:, alpha_i, betas_i * k:(betas_i + 1) * k])

                # concatenating increments size at that dimension and automatically adjusts the indices
                sparse_matrix = torch.cat([sparse_matrix, k_rows])

                del k_rows

            # write the indices and values in storage

        logging.debug(f'rotation {rotation} completed')
        self.write_iv_to_storage(sparse_matrix, rotation)
        del sparse_matrix

    def create_intercept_rows(self, rots, k, dtype=torch.float32):
        """
        stores few rows of the intercept matrix
        indexing is assumed as follows
        for each row (x, y, z).flatten()
        and rows are (phis, alphas, betas).flatten()
        """

        phis = torch.arange(self.p) * self.phi
        all_rays = self.generate_rays(phis, dtype=dtype)
        gc.collect()
        torch.cuda.empty_cache()

        for rotation in tqdm(rots, desc='Generating matrix'):
            # for a rotation
            self.intercept_matrix_per(rotation, k, all_rays[:, rotation, :, :])
            # clean up memory
            gc.collect()
            torch.cuda.empty_cache()
