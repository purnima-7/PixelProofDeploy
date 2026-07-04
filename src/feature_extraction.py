import os
import cv2
import numpy as np
import pandas as pd

from tqdm import tqdm
from skimage.feature import (
    local_binary_pattern,
    graycomatrix,
    graycoprops
)
from skimage.measure import shannon_entropy
from skimage.feature import local_binary_pattern
from skimage.feature import canny
from skimage.transform import probabilistic_hough_line

from PIL import Image
from pillow_heif import register_heif_opener

register_heif_opener()

def load_image(path):

    ext = path.lower().split(".")[-1]

    if ext in ["heic", "heif"]:

        img = Image.open(path).convert("RGB")

        return cv2.cvtColor(
            np.array(img),
            cv2.COLOR_RGB2BGR
        )

    return cv2.imread(path)


# -----------------------------
# CONFIG
# -----------------------------
DATASET_PATH = "../dataset"

# LBP Parameters
RADIUS = 3
N_POINTS = 8 * RADIUS


# -----------------------------
# Feature Functions
# -----------------------------

def laplacian_variance(gray):
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def edge_density(gray):
    edges = cv2.Canny(gray, 100, 200)
    return np.sum(edges > 0) / edges.size


def brightness(gray):
    return np.mean(gray)


def contrast(gray):
    return np.std(gray)


def entropy(gray):
    return shannon_entropy(gray)


def glare_percentage(gray):
    return np.sum(gray > 240) / gray.size


def fft_peak_features(gray):

    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)

    magnitude = np.log(np.abs(fshift) + 1)

    h, w = magnitude.shape
    cy, cx = h // 2, w // 2

    magnitude[cy-20:cy+20, cx-20:cx+20] = 0

    thresh = np.mean(magnitude) + 3*np.std(magnitude)

    peaks = magnitude > thresh

    peak_count = np.sum(peaks)

    peak_strength = magnitude[peaks].sum() if peak_count else 0

    return peak_count, peak_strength

def extract_lbp_features(gray):
    radius = 2
    n_points = 8 * radius

    lbp = local_binary_pattern(
        gray,
        n_points,
        radius,
        method='uniform'
    )

    hist, _ = np.histogram(
        lbp.ravel(),
        bins=np.arange(0, n_points + 3),
        range=(0, n_points + 2)
    )

    hist = hist.astype(np.float32)
    hist /= (hist.sum() + 1e-6)

    return hist

def glcm_features(gray):

    gray = (gray / 8).astype(np.uint8)

    glcm = graycomatrix(
        gray,
        distances=[1],
        angles=[0],
        levels=32,
        symmetric=True,
        normed=True
    )

    contrast = graycoprops(glcm, 'contrast')[0,0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0,0]
    energy = graycoprops(glcm, 'energy')[0,0]
    correlation = graycoprops(glcm, 'correlation')[0,0]

    return contrast, homogeneity, energy, correlation

def rgb_features(img):

    b, g, r = cv2.split(img)

    return (
        np.mean(r),
        np.mean(g),
        np.mean(b),

        np.std(r),
        np.std(g),
        np.std(b)
    )

def hsv_features(img):

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    h, s, v = cv2.split(hsv)

    return (
        np.mean(h),
        np.mean(s),
        np.mean(v),

        np.std(h),
        np.std(s),
        np.std(v)
    )

def pixel_grid_score(gray):

    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1)

    spectrum_x = np.abs(np.fft.fft(gx.mean(axis=0)))
    spectrum_y = np.abs(np.fft.fft(gy.mean(axis=1)))

    score = (
        np.max(spectrum_x[5:]) +
        np.max(spectrum_y[5:])
    )

    return score

def edge_features(gray):
    """
    Returns:
    edge_density
    num_lines
    avg_line_length
    """

    edges = canny(gray / 255.0, sigma=1.2)

    edge_density = np.mean(edges)

    lines = probabilistic_hough_line(
        edges,
        threshold=10,
        line_length=25,
        line_gap=3
    )

    num_lines = len(lines)

    if num_lines:
        lengths = [
            np.sqrt((p0[0]-p1[0])**2 + (p0[1]-p1[1])**2)
            for p0, p1 in lines
        ]
        avg_length = np.mean(lengths)
    else:
        avg_length = 0

    return {
        "edge_density": edge_density,
        "num_lines": num_lines,
        "avg_line_length": avg_length
    }

def fft_peak_ratio(gray):

    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)

    magnitude = np.log(np.abs(fshift) + 1)

    h, w = magnitude.shape
    cy, cx = h // 2, w // 2

    magnitude[cy-15:cy+15, cx-15:cx+15] = 0

    peak = magnitude.max()
    mean = magnitude.mean()

    return {
        "fft_peak_ratio": peak / (mean + 1e-6),
        "fft_peak": peak
    }

def extract_features_from_image(path, label=None):

    img = load_image(path)

    if img is None:
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    gray = cv2.resize(gray, (256, 256))
    img = cv2.resize(img, (256, 256))

    edge = edge_features(gray)
    fft = fft_peak_ratio(gray)

    peak_count, peak_strength = fft_peak_features(gray)

    r_mean, g_mean, b_mean, r_std, g_std, b_std = rgb_features(img)

    h_mean, s_mean, v_mean, h_std, s_std, v_std = hsv_features(img)

    glcm_contrast, glcm_homogeneity, glcm_energy, glcm_correlation = glcm_features(gray)

    features = {

        "laplacian_variance": laplacian_variance(gray),
        "edge_density": edge["edge_density"],
        "brightness": brightness(gray),
        "contrast": contrast(gray),
        "entropy": entropy(gray),
        "glare_percentage": glare_percentage(gray),

        "pixel_grid_score": pixel_grid_score(gray),

        "num_lines": edge["num_lines"],
        "avg_line_length": edge["avg_line_length"],

        "fft_peak_ratio": fft["fft_peak_ratio"],
        "fft_peak": fft["fft_peak"],

        "fft_peak_count": peak_count,
        "fft_peak_strength": peak_strength,

        "glcm_contrast": glcm_contrast,
        "glcm_homogeneity": glcm_homogeneity,
        "glcm_energy": glcm_energy,
        "glcm_correlation": glcm_correlation,

        "r_mean": r_mean,
        "g_mean": g_mean,
        "b_mean": b_mean,

        "r_std": r_std,
        "g_std": g_std,
        "b_std": b_std,

        "h_mean": h_mean,
        "s_mean": s_mean,
        "v_mean": v_mean,

        "h_std": h_std,
        "s_std": s_std,
        "v_std": v_std,
    }

    lbp_hist = extract_lbp_features(gray)

    for i, value in enumerate(lbp_hist):
        features[f"lbp_{i}"] = value

    if label is not None:
        features["image"] = os.path.basename(path)
        features["label"] = label

    return features


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    rows = []

    for label in ["real", "screen"]:

        folder = os.path.join(DATASET_PATH, label)

        for file in tqdm(os.listdir(folder), desc=label):
        
            path = os.path.join(folder, file)

            features = extract_features_from_image(path, label)

            if features is not None:
                rows.append(features)


    df = pd.DataFrame(rows)

    df.to_csv("../features.csv", index=False)

    print()
    print("=" * 50)
    print(df.head())
    print("=" * 50)
    print()
    print("Saved as features.csv")