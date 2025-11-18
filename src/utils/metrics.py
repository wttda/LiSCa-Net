import numpy as np
from scipy.signal import convolve2d
from scipy.ndimage import gaussian_filter
from sklearn.metrics import mean_squared_error


def MPSNR(hsi, ref):
    """Calculate the Mean Peak Signal-to-Noise Ratio (MPSNR) between two HSIs.

    Parameters:
        hsi (2D np.ndarray): Array of shape (band, row * column).
        ref (2D np.ndarray): Reference array of shape (band, row * column).

    Returns (float): MPSNR.
    """
    assert hsi.ndim == 2 and hsi.shape == ref.shape
    mse = np.mean((hsi - ref) ** 2, axis=1)
    psnr = np.where(mse == 0, np.inf, 10 * np.log10(1.0 / mse))
    return np.mean(psnr)

def gaussian_window(size, sigma):
    """Generate a Gaussian window of given size and sigma.

    Parameters:
        size (int): Size of the Gaussian window.
        sigma (float): Standard deviation of the Gaussian distribution.

    Returns (np.ndarray): Gaussian window.
    """
    coords = np.arange(-size // 2 + 1, size // 2 + 1)
    x, y = np.meshgrid(coords, coords)
    g = np.exp(-(x ** 2 + y ** 2) / (2.0 * sigma ** 2))
    return g / np.sum(g)


def compute_ssim(img1, img2, K=None, window=None, L=255):
    """Calculate the Structural Similarity (SSIM) index between two images.

    Parameters:
        img1 (np.ndarray): The first image being compared.
        img2 (np.ndarray): The second image being compared.
        K (list, optional): Constants in the SSIM index formula. Default: [0.01, 0.03].
        window (numpy.ndarray, optional): Local window for statistics.
            Default: Gaussian window of size 11x11 and sigma 1.5.
        L (int, optional): Dynamic range of the images. Default: 255.

    Returns:
        mssim (float): The mean SSIM index value between 2 images.
        ssim_map (np.ndarray): The SSIM index map of the test image.
    """
    if img1.shape != img2.shape:
        raise ValueError("Input images must have the same dimensions.")
    M, N = img1.shape
    if K is None:
        K = [0.01, 0.03]
    if window is None:
        window = gaussian_filter(np.ones((11, 11)), 1.5)
    else:
        H, W = window.shape
        if H * W < 4 or H > M or W > N:
            raise ValueError("Window size is invalid.")

    window = window / np.sum(window)
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mu1 = convolve2d(img1, window, mode='valid')
    mu2 = convolve2d(img2, window, mode='valid')
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = convolve2d(img1 ** 2, window, mode='valid') - mu1_sq
    sigma2_sq = convolve2d(img2 ** 2, window, mode='valid') - mu2_sq
    sigma12 = convolve2d(img1 * img2, window, mode='valid') - mu1_mu2
    C1 = (K[0] * L) ** 2
    C2 = (K[1] * L) ** 2

    # Calculate SSIM
    if C1 > 0 and C2 > 0:
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    else:
        numerator1 = 2 * mu1_mu2 + C1
        numerator2 = 2 * sigma12 + C2
        denominator1 = mu1_sq + mu2_sq + C1
        denominator2 = sigma1_sq + sigma2_sq + C2
        ssim_map = np.ones_like(mu1)
        index = (denominator1 * denominator2 > 0)
        ssim_map[index] = (numerator1[index] * numerator2[index]) / (denominator1[index] * denominator2[index])
        index = (denominator1 != 0) & (denominator2 == 0)
        ssim_map[index] = numerator1[index] / denominator1[index]
    return np.mean(ssim_map), ssim_map


def MSSIM(hsi, ref, rows, cols):
    """Calculate the Mean Structural Similarity Index (MSSIM) between two arrays.

    Parameters:
        hsi (np.ndarray): Array of shape (band, row * column).
        ref (np.ndarray): Reference array of shape (band, row * column).

    Returns (float): MSSIM.
    """
    K, window = [0.01, 0.03], gaussian_window(11, 1.5)
    mssim_vals = []
    for hsi_band, ref_band in zip(hsi, ref):
        L = max(np.max(hsi_band), np.max(ref_band))
        ref_img = ref_band.reshape(rows, cols, order='F')
        hsi_img = hsi_band.reshape(rows, cols, order='F')
        mssim_vals.append(compute_ssim(ref_img, hsi_img, K, window, L)[0])
    return np.mean(mssim_vals)

def ERGAS(hsi, ref, rows, cols):
    """Calculate the Error Relative Global Dimensionless Synthesis (ERGAS) between two HSIs.

    Parameters:
        hsi (np.ndarray): Array of shape (band, row * column).
        ref (np.ndarray): Reference array of shape (band, row * column).
        rows (int): Number of rows in the HSI.
        cols (int): Number of columns in the HSI.

    Returns (float): ERGAS value.
    """
    assert hsi.ndim == 2 and hsi.shape == ref.shape
    nbd, _ = ref.shape
    hsi = hsi.reshape(nbd, rows, cols)
    ref = ref.reshape(nbd, rows, cols)
    ergas_num = sum(mean_squared_error(ref[i], hsi[i]) / (np.mean(ref[i]) ** 2) for i in range(nbd))
    return 100 * np.sqrt(ergas_num / nbd)
