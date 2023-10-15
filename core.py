import gc
import torch
from tqdm import tqdm
import logging

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class CreateInterceptMatrix:
    def __init__(self, detector_plate_length, source_to_object, source_to_detector, pixel_size, projections,
                 dtype=torch.float16, resolution=None):
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
            float16 gives bad precision for larger ps value, use it only if the whole space is being scaled
            because float16 operations are faster than float32 on GPU
        """
        self.dl = detector_plate_length
        # deliberatly scaling my space by factor of 1/100 so that precision in float16 is better
        self.sod = source_to_object / 100
        self.sdd = source_to_detector / 100
        self.ps = pixel_size / 100
        self.p = projections

        # Assumption: maximum angle is 2pi
        self.phi = 2 * torch.pi / projections
        self.n = resolution if resolution is not None else detector_plate_length
        assert self.n % 2 == 0
        self.dtype = dtype

    @staticmethod
    def write_iv_to_storage(sparse_matrix, rot_no):
        # write to 2 files indices and values
        torch.save(sparse_matrix, f'matrix_rot_{rot_no}.pt')
        del sparse_matrix

    def intercepts_for_rays(self, ray_coords, dtype=None):
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
        sod = self.sod

        # creating the grid of voxels
        a = torch.arange((-n // 2) * ps, (n // 2) * ps, ps, device=device, dtype=dtype)
        X, Y, Z = torch.meshgrid(a, a, a, indexing='xy')

        # flatten
        X = X.flatten().view(1, -1)
        Y = Y.flatten().view(1, -1)
        Z = Z.flatten().view(1, -1)

        # slopes in parametric form of the 3d ray
        x_line_slope = sod * torch.sin(phi) - alpha
        z_line_slope = sod * torch.cos(phi)

        # Defining equations which will generate rays acc to the parametric form
        def xy_from_z(z):
            return (alpha + z * x_line_slope / z_line_slope,
                    betas - z * betas / z_line_slope)

        def yz_from_x(x):
            return (betas - betas * (x - alpha) / x_line_slope,
                    (x - alpha) * z_line_slope / x_line_slope)

        def zx_from_y(y):
            return (-z_line_slope * (y - betas) / betas,
                    alpha - x_line_slope * (y - betas) / betas)

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

        del X, Y, Z, alpha, betas, phi, x_line_slope, z_line_slope

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

    def generate_rays(self, phis, dtype=torch.float16):
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

        mu = self.sdd - self.sod
        lambd = self.sod
        c = torch.cos(phis)
        s = torch.sin(phis)
        a = detector_coords[0]
        b = detector_coords[1]
        alphas = (a * lambd + lambd * mu * s) / (a * s + mu + lambd * c ** 2)
        betas = b / (1 - (mu + alphas * s) / (alphas * s - lambd))

        phis = phis + torch.zeros_like(alphas)
        line_params_tensor = torch.stack([phis, alphas, betas], 0)

        del phis, alphas, betas, a, b, detector_coords, c, s, x
        gc.collect()
        torch.cuda.empty_cache()
        return line_params_tensor.to('cpu')

    def intercept_matrix_per(self, rotation, k, all_rays_rot, dtype=torch.float16):
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

        # full size of the final intercept matrix
        full_size = torch.Size([self.p * self.dl * self.dl, self.n ** 3])
        sparse_matrix = torch.sparse_coo_tensor(full_size, dtype=dtype)

        for alpha_i in tqdm(range(self.n), leave=False):
            for betas_i in range(self.n // k):
                # peak GPU RAM usage
                k_rows = self.intercepts_for_rays(all_rays_rot[:, alpha_i, betas_i*k:(betas_i+1)*k])
                new_indices = k_rows.indices()

                new_indices[0] = new_indices[0] + rotation*self.n*self.n + alpha_i*self.n + betas_i*k

                sparse_matrix = torch.cat([sparse_matrix, torch.sparse_coo_tensor(new_indices, k_rows.values(), full_size, dtype=dtype)])

                del k_rows, new_indices

            # write the indices and values in storage

        logging.debug(f'rotation {rotation} completed, created matrix of size {len(sparse_matrix.values())}')
        self.write_iv_to_storage(sparse_matrix, rotation)
        del sparse_matrix

    def create_intercept_rows(self, rot_start, rot_end, k, dtype=torch.float16):
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

        for rotation in tqdm(range(rot_start, rot_end), desc='Generating matrix'):
            # for a rotation
            self.intercept_matrix_per(rotation, k, all_rays[:, rotation, :, :])
            # clean up memory
            gc.collect()
            torch.cuda.empty_cache()
