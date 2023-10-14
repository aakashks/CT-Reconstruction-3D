import gc
import torch
from tqdm import tqdm

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
            float16 gives bad precision. error of 1mm can occur. prefer using float32 unless facing gpu ram shortage
        """
        self.dl = detector_plate_length
        self.sod = source_to_object
        self.sdd = source_to_detector
        self.ps = pixel_size
        self.p = projections

        # Assumption: maximum angle is 2pi
        self.phi = 2 * torch.pi / projections
        self.n = resolution if resolution is not None else detector_plate_length
        assert self.n % 2 == 0
        self.dtype = dtype

    @staticmethod
    def write_iv_to_storage(indices, values, rot_no):
        # write to 2 files indices and values
        torch.save(indices, f'indices_rot{rot_no}.pt')
        torch.save(values, f'values_rot{rot_no}.pt')

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
        a = torch.arange(-n // 2, n // 2, 1, device=device, dtype=dtype) * ps
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

    def generate_rays(self, phis, dtype=torch.float32):
        """
        generate rays from source to each detector for a rotation angle of phi
        returns shape (3, phis, alphas, betas)
        where alphas = betas = dl = n
        """
        n = self.dl
        ps = self.ps
        x = torch.arange(-n + 1, n, 2, device=device, dtype=dtype) / 2 * ps
        detector_coords = torch.stack(torch.meshgrid(x, x, indexing='xy'), 0)

        phis = phis.to(device, dtype).reshape(-1, 1, 1)

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

    def intercept_matrix_per(self, rotation, k, all_rays_rot):
        """
        for only 1 rotation
        write to storage sparse tensor of shape (alphas, betas, x, y, z)
        x = y = z = n
        with approx n elements for each (alphas, betas, :)

        all_rays_rot
            all rays for 1 rotation. shape (3, alphas, betas)

        k
            no of betas together
            a suitable value for n=200 can be 100 for colab, 200 for kaggle (16GBs). 50 for 8GB
        """
        assert self.n % k == 0

        # store indices and values of sparse matrix
        indices = torch.empty(size=[2, 0])
        values = torch.empty(size=[0])

        for alpha_i in tqdm(range(self.n), leave=False):
            for betas_i in range(self.n // k):
                # peak GPU RAM usage
                k_rows = self.intercepts_for_rays(all_rays_rot[:, alpha_i, betas_i*k:(betas_i+1)*k])
                new_indices = k_rows.indices()

                new_indices[0] = new_indices[0] + rotation*self.n*self.n + alpha_i*self.n + betas_i*k
                indices = torch.cat([indices, new_indices], dim=1)
                values = torch.cat([values, k_rows.values()])

                del k_rows, new_indices

            # write the indices and values in storage
        self.write_iv_to_storage(indices, values, rotation)

    def create_intercept_rows(self, rot_start, rot_end, k, dtype=torch.float32):

        phis = torch.arange(self.p) * self.phi
        all_rays = self.generate_rays(phis, dtype=dtype)
        gc.collect()
        torch.cuda.empty_cache()

        for rotation in tqdm(range(rot_start, rot_end), desc='Generating matrix: rotation-'):
            # for a rotation
            self.intercept_matrix_per(rotation, k, all_rays[:, rotation, :, :])
            # clean up memory
            gc.collect()
            torch.cuda.empty_cache()
