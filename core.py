import torch
import gc
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class CreateInterceptMatrix:
    def __init__(self, detector_plate_length, source_to_object, source_to_detector, pixel_size, projections,
                 resolution=None):
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
        resolution
            ASSUMING same resolution along all axis
        """
        self.dl = detector_plate_length
        self.sod = source_to_object
        self.sdd = source_to_detector
        self.ps = pixel_size
        self.p = projections

        # Assumption: maximum angle is 2pi
        self.phi = 2 * torch.pi / projections
        self.n = resolution if resolution is not None else detector_plate_length

    @staticmethod
    def write_iv_to_storage(indices, values):
        # write to 2 files indices and values
        pass

    def intercepts_for_ray(self, ray_coord, phi):
        """
        get all voxel intercepts for 1 ray
        line parameters are taken using centre of object as origin
        """
        alpha, beta = ray_coord
        alpha = alpha.item()
        beta = beta.item()

        # each pixel is represented by its bottom left corner coordinate
        ps = self.ps
        n = self.n
        sod = self.sod

        # creating the grid of voxels
        x = torch.arange(-n // 2, n // 2, 1) if n % 2 == 0 else torch.arange(-n, n + 1, 2) / 2
        x = x.to(torch.float16) * ps
        X, Y, Z = torch.meshgrid(x, x, x, indexing='ij')

        phi = torch.tensor(phi)

        # line slopes
        x_line_slope = sod * torch.sin(phi) - alpha
        z_line_slope = sod * torch.cos(phi)

        # Defining equations which will generate lines
        def xy_from_z(z):
            z = z.to(device)
            x = alpha + z * x_line_slope / z_line_slope
            y = beta - z * beta / z_line_slope
            return x.to('cpu'), y.to('cpu')

        def yz_from_x(x):
            x = x.to(device)
            y = beta - beta * (x - alpha) / x_line_slope
            z = (x - alpha) * z_line_slope / x_line_slope
            return y.to('cpu'), z.to('cpu')

        def zx_from_y(y):
            y = y.to(device)
            z = -z_line_slope * (y - beta) / beta
            x = alpha - x_line_slope * (y - beta) / beta
            return z.to('cpu'), x.to('cpu')

        # Get line intercepts with 6 boundaries of each voxel
        # TODO: cuda RAM management
        X1, Y1 = xy_from_z(Z)
        X2, Y2 = xy_from_z(Z + 1)

        Y3, Z3 = yz_from_x(X)
        Y4, Z4 = yz_from_x(X + 1)

        Z5, X5 = zx_from_y(Y)
        Z6, X6 = zx_from_y(Y + 1)

        # Create masks for the conditions
        mask1 = (X <= X1) & (X1 < X + 1) & (Y <= Y1) & (Y1 < Y + 1)
        mask2 = (X <= X2) & (X2 < X + 1) & (Y <= Y2) & (Y2 < Y + 1)
        mask3 = (Y <= Y3) & (Y3 < Y + 1) & (Z <= Z3) & (Z3 < Z + 1)
        mask4 = (Y <= Y4) & (Y4 < Y + 1) & (Z <= Z4) & (Z4 < Z + 1)
        mask5 = (X <= X5) & (X5 < X + 1) & (Z <= Z5) & (Z5 < Z + 1)
        mask6 = (X <= X6) & (X6 < X + 1) & (Z <= Z6) & (Z6 < Z + 1)

        I1 = torch.stack([mask1 * X1, mask1 * Y1, mask1 * Z], 3)
        I2 = torch.stack([mask2 * X2, mask2 * Y2, mask2 * (Z + 1)], 3)
        I3 = torch.stack([mask3 * X, mask3 * Y3, mask3 * Z3], 3)
        I4 = torch.stack([mask4 * (X + 1), mask4 * Y4, mask4 * Z4], 3)
        I5 = torch.stack([mask5 * X5, mask5 * Y, mask5 * Z5], 3)
        I6 = torch.stack([mask6 * X6, mask6 * (Y + 1), mask6 * Z6], 3)

        # To get length of line from all these intercept coordinates
        intercept_coordinates = torch.abs(torch.abs(torch.abs(I1 - I2) - torch.abs(I3 - I4)) - torch.abs(I5 - I6))

        # now squaring will give the length
        intercept_matrix = torch.linalg.norm(intercept_coordinates, dim=3)

        # change to 1d vector
        # return intercept_matrix.to_sparse_coo().to(device='cpu', dtype=torch.float16)

    def generate_rays(self, phi):
        """
        generate rays from source to each detector for a rotation angle of phi
        """
        n = self.dl
        p = self.ps
        x = torch.arange(-n + 1, n, 2) / 2 * p if n % 2 == 0 else torch.arange((-n + 1) // 2, (n + 1) // 2) * p
        x = x.to(device, torch.float32)
        detector_coords = torch.dstack(torch.meshgrid(x, x)).reshape(-1, 2)

        mu = self.sdd - self.sod
        lambd = self.sod
        c = torch.cos(phi)
        s = torch.sin(phi)
        a = detector_coords[:, 0:1]
        b = detector_coords[:, 1:2]
        alphas = (a * lambd + lambd * mu * s) / (a * s + mu + lambd * c ** 2)
        betas = b.reshape(1, -1) / (1 - (mu + alphas * s) / (alphas * s - lambd))

        line_params_tensor = torch.stack([alphas, betas], 2).reshape(-1, 2)

        return line_params_tensor.to('cpu')

    def intercept_matrix_per(self, rotation):
        angle = rotation * self.phi
        n2 = self.n * self.n
        all_rays_for_1_rotation = self.generate_rays(angle)  # (n*n, 2)

        # store indices and values of sparse matrix
        indices = torch.empty(size=[2, 0])
        values = torch.empty(size=[0])

        for i in range(n2):
            # sparse vector which is 1 row of intercept matrix
            intercept_row_for_each_ray_sparse = self.intercepts_for_ray(all_rays_for_1_rotation[i], angle)  # (1, -)

            # according to i and rotation, decide the indices[0] and concatenate it
            new_indices = intercept_row_for_each_ray_sparse.indices()
            new_indices[0] = i + n2 * rotation

            indices = torch.cat([indices, new_indices], dim=1)
            values = torch.cat([values, intercept_row_for_each_ray_sparse.values()])

            del intercept_row_for_each_ray_sparse, new_indices

        # write the indices and values in storage
        self.write_iv_to_storage(indices, values)

    def create_intercept_matrix_from_lines(self):

        for rotation in tqdm(range(self.p), desc='Generating matrix: rotation-'):
            # for a rotation
            self.intercept_matrix_per(rotation)
            # clean up memory
            torch.cuda.empty_cache()
            gc.collect()
