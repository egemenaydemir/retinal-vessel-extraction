"""
Retinal Blood Vessel Extraction Pipeline
----------------------------------------
Description: Implementation of a hybrid method combining supervised and 
unsupervised ML for retinal vessel segmentation.

Author: Egemen Aydemir
Affiliation: Izmir Institute of Technology (IZTECH)

Reference Paper:
"Retinal blood vessel extraction employing effective image features and 
combination of supervised and unsupervised machine learning methods"
Authors: Mahdi Hashemzadeh, Baharak Adlpour Azar 
Journal: Artificial Intelligence In Medicine, 2019
DOI: https://doi.org/10.1016/j.artmed.2019.03.001 
"""

import csv
import cv2
import numpy as np 
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesClassifier
import skfuzzy as fuzz
import os
import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# ========================== UTILITY FUNCTIONS ==========================

def load_image(image_path):
    """Load an image from the specified path."""
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Image not found: {image_path}")
    return image

def load_mask(mask_path):
    """Load a mask image and convert to binary."""
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Mask not found: {mask_path}")
    _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    return binary_mask

def crop_to_fov(image, mask):
    """Crop the image to the field of view bounding box."""
    y_indices, x_indices = np.where(mask > 0)
    y_min, y_max = y_indices.min(), y_indices.max()
    x_min, x_max = x_indices.min(), x_indices.max()
    
    #if image has color channels
    if len(image.shape) == 3: 
        return image[y_min:y_max+1, x_min:x_max+1, :], (y_min, y_max, x_min, x_max)
    
    return image[y_min:y_max+1, x_min:x_max+1], (y_min, y_max, x_min, x_max)

def crop_mask(mask, bounds):
    """Crop mask using the same bounds."""
    y_min, y_max, x_min, x_max = bounds
    return mask[y_min:y_max+1, x_min:x_max+1]

def fill_outside_fov(image, mask, fill_value):
    """Fill pixels outside FOV with a specified value."""
    filled_image = image.copy()
    filled_image[mask == 0] = fill_value
    return filled_image

def invert_image(image):
    """Invert grayscale image."""
    if image.dtype != np.uint8:
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return 255 - image

# ========================== PREPROCESSING ==========================

def extract_channels(image):
    """Extract G, Y, L channels from the image."""
    G = image[:, :, 1]  # G from BGR
    ycbcr = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    Y = ycbcr[:, :, 0]  # Y from YCbCr
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    L = lab[:, :, 0]  # L from L*a*b
    return G, Y, L

def apply_clahe(channel, clip_limit=2.0, tile_size=(8, 8)):
    """Apply CLAHE enhancement."""
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
    return clahe.apply(channel.astype(np.uint8))

def preprocess_image(image, mask):
    """Complete preprocessing pipeline."""
    #Step1 - Crop to FOV
    cropped_image, bounds = crop_to_fov(image, mask)
    cropped_mask = crop_mask(mask, bounds)

    #Step2 - Extract channels
    G, Y, L = extract_channels(cropped_image)

    #Step3 - fill with median value outside FOV to avoid artifacts during CLAHE
    G = fill_outside_fov(G, cropped_mask, fill_value=int(np.median(G[cropped_mask > 0])))
    Y = fill_outside_fov(Y, cropped_mask, fill_value=int(np.median(Y[cropped_mask > 0])))
    L = fill_outside_fov(L, cropped_mask, fill_value=int(np.median(L[cropped_mask > 0])))

    #Step4 - Apply CLAHE
    G_clahe = apply_clahe(G)
    Y_clahe = apply_clahe(Y)
    L_clahe = apply_clahe(L)

    #Step5 - Fill outside FOV with 0
    G_clahe = fill_outside_fov(G_clahe, cropped_mask, fill_value=0)
    Y_clahe = fill_outside_fov(Y_clahe, cropped_mask, fill_value=0)
    L_clahe = fill_outside_fov(L_clahe, cropped_mask, fill_value=0)
    
    return {
        'G_raw': G,
        'Y_raw': Y,
        'L_raw': L,
        'G_clahe': G_clahe,
        'Y_clahe': Y_clahe,
        'L_clahe': L_clahe,
        'cropped_mask': cropped_mask,
        'cropped_image': cropped_image,
        'bounds': bounds
    }
# ========================= GABOR FILTERING ==========================

def compute_gabor_sigma(wavelength, bandwidth):
    """Compute sigma for Gabor filter."""
    return (wavelength / np.pi) * np.sqrt(np.log(2) / 2) * ((2**bandwidth + 1) / (2**bandwidth - 1))

def create_gabor_kernel(wavelength, theta, gamma, bandwidth, psi):    
    """Create a Gabor kernel. """

    sigma = compute_gabor_sigma(wavelength, bandwidth)
    ksize = int(np.ceil(6 * sigma))
    if ksize % 2 == 0:
        ksize += 1  # Ensure ksize is odd

    half_size = ksize // 2
    y, x = np.meshgrid(range(-half_size, half_size + 1), range(-half_size, half_size + 1))

    x_theta = x * np.cos(theta) + y * np.sin(theta)
    y_theta = -x * np.sin(theta) + y * np.cos(theta)

    gb = np.exp(-0.5 * (x_theta**2 + (gamma**2) * y_theta**2) / (sigma**2)) * np.cos(2 * np.pi * x_theta / wavelength + psi)
    return gb

def gabor_filter_response(channel, wavelength, gamma=0.5, bandwidth=1, psi=0): 
    """
    Apply Gabor filter to the channel.

    inverted channel is used so vessels appear bright. (Not mentioned in the paper)
    """
    img_inverted = invert_image(channel)
    img_inverted = img_inverted.astype(np.float64)
    
    responses = []
    for angle in range(0, 180, 15):
        theta = angle * np.pi / 180 
        
        kernel = create_gabor_kernel(wavelength, theta, gamma, bandwidth, psi)
        
        filtered = cv2.filter2D(img_inverted, cv2.CV_64F, kernel)
        responses.append(filtered)
    
    max_response = np.max(responses, axis=0)

    max_response = np.maximum(0, max_response)  # Set negative values to zero
    
    return max_response

def extract_gabor_features(G, Y, L, wavelengths=[9, 10, 11]): 
    """
    Extract 9 Gabor features (3 channels × 3 wavelengths) for an image. 
    """
    gabor_features = []
    for channel in [G, Y, L]:
        for wavelength in wavelengths:
            response = gabor_filter_response(channel, wavelength)  
            gabor_features.append(response)

    return gabor_features

# ========================= AUTOMATIC THRESHOLDING ==========================

def calculate_automatic_threshold(image):
    """
    Calculate automatic threshold using paper's formula:
    T = Σ(i × Yi/(m×n)) for i=0 to 255
    """
    total_pixels = image.shape[0] * image.shape[1]
    hist = cv2.calcHist([image], [0], None, [256], [0, 256]).flatten()
    threshold = 0.0
    for i in range(256):
        threshold += i * (hist[i] / total_pixels)
    
    return threshold

def binarize_gabor_features(gabor_feature, mask, method):
    """
    Binarize Gabor features.
    
    Laplacian enhancement + automatic threshold (paper method)
    """
    gabor_norm = cv2.normalize(gabor_feature, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    if method == 'otsu':
        _, binary = cv2.threshold(gabor_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        laplacian = cv2.Laplacian(gabor_norm, cv2.CV_64F)
        laplacian = laplacian.astype(np.uint8)
        combined = (laplacian + gabor_norm).astype(np.uint8)
        
        # Calculate automatic threshold on combined image (using mask)
        threshold = calculate_automatic_threshold(combined)
        _, binary = cv2.threshold(combined, int(threshold), 255, cv2.THRESH_BINARY)
        
    binary = fill_outside_fov(binary, mask, fill_value=0)
    return binary

# ========================= TOP-HAT EXTRACTION ==========================

def create_line_kernel(length, angle):
    """Create a line structuring element."""
    angle_rad = np.deg2rad(angle)
    x = int((length - 1) * np.cos(angle_rad))
    y = int((length - 1) * np.sin(angle_rad))
    kernel = np.zeros((length, length), dtype=np.uint8)
    cv2.line(kernel, (length // 2 - x // 2, length // 2 - y // 2), (length // 2 + x // 2, length // 2 + y // 2), 1, thickness=1) # Draw line in the kernel
    return kernel

def top_hat_extraction(image):
    inverted_G_clahe = invert_image(image)

    angles = np.arange(0,180,22.5) #nine angles from 0 to 157.5
    structuring_elements = []
    for angle in angles:
        kernel = create_line_kernel(length=21, angle=angle)
        structuring_elements.append(kernel)

    oppened_images = []
    for se in structuring_elements:
        opened = cv2.morphologyEx(inverted_G_clahe, cv2.MORPH_OPEN, se)
        oppened_images.append(opened)

    top_hat_images = []
    for opened in oppened_images:
        top_hat = cv2.subtract(inverted_G_clahe, opened)
        top_hat_images.append(top_hat)

    #take pixel-wise maximum of opened images
    max_top_hat = np.maximum.reduce(top_hat_images)

    return max_top_hat

# ========================= SHADE CORRECTED FEATURE EXTRACTION ==========================

def SC_extraction(image):
    G_clahe_median = cv2.medianBlur(image, 25)

    diff = G_clahe_median.astype(np.float32) - image.astype(np.float32)
    SC_normalized = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)
    SC_feature = SC_normalized.astype(np.uint8)

    return SC_feature                                                                                                                                                                                                                                           

# ========================= FEATURE VECTOR CREATION ==========================

def create_feature_vector(gabor_features, G_clahe, th_feature, sc_feature):
    """
    Create feature vector by stacking
    Gabor features, TH feature, and SC feature for one image.
    """
    feature_vector = np.stack([G_clahe] + gabor_features + [th_feature, sc_feature], axis=-1)
    feature_vector = feature_vector.reshape(-1, feature_vector.shape[-1])
    return feature_vector

# ========================= FUZZY C-MEANS CLUSTERING ==========================

def fcm_fit_predict(X, n_clusters=2, m=2.0, error=1e-6, maxiter=2000, seed=40):
    """
    X: (n_samples, n_features)
    return:
      labels: (n_samples,)
      U: (n_clusters, n_samples) membership
      centers: (n_clusters, n_features)
    """
    rng = np.random.default_rng(seed)
    # Expected format: (n_features, n_samples)
    data = X.T.astype(np.float64)

    cntr, U, U0, d, jm, p, fpc = fuzz.cluster.cmeans(
        data=data,
        c=n_clusters,
        m=m,
        error=error,
        maxiter=maxiter,
        init=None,          # if desired, U0 can be provided
        seed=seed
    )

    labels = np.argmax(U, axis=0)
    return labels, U, cntr, fpc

def find_cluster_with_less_pixels(labels):
    """Find the cluster index that has less pixels (assumed to be vessel cluster)."""
    unique, counts = np.unique(labels, return_counts=True)
    cluster_counts = dict(zip(unique, counts))
    vessel_cluster = min(cluster_counts, key=cluster_counts.get)
    return vessel_cluster

def binary_fcm_prediction(labels, vessel_cluster):
    """Convert FCM labels to binary prediction (vessel vs non-vessel)."""
    binary_prediction = (labels == vessel_cluster).astype(np.uint8) * 255
    return binary_prediction

# ========================= POST-PROCESSING ==========================

def remove_fov_ring_only(classified_img, fov_img, border_px=12, outside_zero=True):
    """
    Remove only the border ring of the FOV from the classified image.
    Parameters:
    - classified_img: The input classified image (grayscale or color).
    - fov_img: The FOV mask image (grayscale or color).
    - border_px: The width of the border ring to remove (in pixels).
    - outside_zero: If True, also set pixels outside the FOV to 0.

    Returns:
    - cleaned: The classified image with only the FOV border ring removed.
    """

    #gray
    pred = cv2.cvtColor(classified_img, cv2.COLOR_BGR2GRAY) if classified_img.ndim == 3 else classified_img.copy()
    fov  = cv2.cvtColor(fov_img, cv2.COLOR_BGR2GRAY) if fov_img.ndim == 3 else fov_img.copy()

    #resize FOV to match pred 
    if pred.shape != fov.shape:
        fov = cv2.resize(fov, (pred.shape[1], pred.shape[0]), interpolation=cv2.INTER_NEAREST)

    #binarize
    pred_bin = ((pred > 0).astype(np.uint8)) * 255
    fov_bin  = ((fov  > 0).astype(np.uint8)) * 255

    #Create a structuring element for erosion
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*border_px+1, 2*border_px+1))
    fov_eroded = cv2.erode(fov_bin, k)
    ring = cv2.subtract(fov_bin, fov_eroded)  

    #Only remove the ring pixels from the classified image
    cleaned = pred_bin.copy()
    cleaned[ring > 0] = 0

    #Optionally set pixels outside the FOV to 0
    if outside_zero:
        cleaned[fov_bin == 0] = 0

    return cleaned

# ========================== MAIN PIPELINE ==========================

def load_drive_dataset(data_path):
    """Load DRIVE dataset."""
    print("Loading DRIVE dataset...")
    
    # Training images (21-40)
    train_images, train_masks, train_gt = [], [], []
    for i in range(21, 41):
        img_path = os.path.join(data_path, f"training/images/{i}_training.tif")
        mask_path = os.path.join(data_path, f"training/mask/{i}_training_mask.gif")
        gt_path = os.path.join(data_path, f"training/1st_manual/{i}_manual1.gif")
        
        train_images.append(load_image(img_path))
        train_masks.append(load_mask(mask_path))
        train_gt.append(load_mask(gt_path))
    
    # Test images (01-20)
    test_images, test_masks, test_gt = [], [], []
    for i in range(1, 21):
        img_path = os.path.join(data_path, f"test/images/{i:02d}_test.tif")
        mask_path = os.path.join(data_path, f"test/mask/{i:02d}_test_mask.gif")
        gt_path = os.path.join(data_path, f"test/1st_manual/{i:02d}_manual1.tif")
        if not os.path.exists(gt_path):
            gt_path = os.path.join(data_path, f"test/1st_manual/{i:02d}_manual1.gif")
        
        test_images.append(load_image(img_path))
        test_masks.append(load_mask(mask_path))
        test_gt.append(load_mask(gt_path))
    
    print(f"Loaded {len(train_images)} training and {len(test_images)} test images")
    return train_images, train_masks, train_gt, test_images, test_masks, test_gt

def preprocess_dataset(training_images, training_masks, test_images, test_masks):
    """
    Preprocess entire dataset.
    Save figures 3 and 4 for each image.
    """
    preprocessed_train = {
        'G_raw': [],
        'Y_raw': [],
        'L_raw': [],
        'G_clahe': [],
        'Y_clahe': [],
        'L_clahe': [],
        'cropped_mask': [],
        'cropped_image': [],
        'bounds': []
    }
    preprocessed_test = {
        'G_raw': [],
        'Y_raw': [],
        'L_raw': [],
        'G_clahe': [],
        'Y_clahe': [],
        'L_clahe': [],
        'cropped_mask': [],
        'cropped_image': [],
        'bounds': []
    }
    
    print("Preprocessing train data...")
    for idx, (img, mask) in enumerate(zip(training_images, training_masks)):
        result = preprocess_image(img, mask)
        # Save figures
        save_fig3(result['G_raw'], result['Y_raw'], result['L_raw'], f"results/fig3/train_fig3_{idx+21}.png")
        save_fig4(result['G_clahe'], result['Y_clahe'], result['L_clahe'], f"results/fig4/train_fig4_{idx+21}.png")
        
        preprocessed_train['G_raw'].append(result['G_raw'])
        preprocessed_train['Y_raw'].append(result['Y_raw'])
        preprocessed_train['L_raw'].append(result['L_raw'])
        preprocessed_train['G_clahe'].append(result['G_clahe'])
        preprocessed_train['Y_clahe'].append(result['Y_clahe'])
        preprocessed_train['L_clahe'].append(result['L_clahe'])
        preprocessed_train['cropped_mask'].append(result['cropped_mask'])
        preprocessed_train['cropped_image'].append(result['cropped_image'])
        preprocessed_train['bounds'].append(result['bounds'])
        
    
    print("Preprocessing test data...")
    for idx, (img, mask) in enumerate(zip(test_images, test_masks)):
        result = preprocess_image(img, mask)
        
        # Save figures
        save_fig3(result['G_raw'], result['Y_raw'], result['L_raw'], f"results/fig3/test_fig3_{idx+1:02d}.png")
        save_fig4(result['G_clahe'], result['Y_clahe'], result['L_clahe'], f"results/fig4/test_fig4_{idx+1:02d}.png")
            
        preprocessed_test['G_raw'].append(result['G_raw'])
        preprocessed_test['Y_raw'].append(result['Y_raw'])
        preprocessed_test['L_raw'].append(result['L_raw'])
        preprocessed_test['G_clahe'].append(result['G_clahe'])
        preprocessed_test['Y_clahe'].append(result['Y_clahe'])
        preprocessed_test['L_clahe'].append(result['L_clahe'])
        preprocessed_test['cropped_mask'].append(result['cropped_mask'])
        preprocessed_test['cropped_image'].append(result['cropped_image'])
        preprocessed_test['bounds'].append(result['bounds'])
    
    return preprocessed_train, preprocessed_test
    
def gabor_feature_extraction_dataset(preprocessed_train, preprocessed_test):
    """
    Extract Gabor features for entire dataset.
    Save figure 5 for each image.
    """
    gabor_train = {
        'Gabor_G9': [],
        'Gabor_G10': [],
        'Gabor_G11': [],
        'Gabor_Y9': [],
        'Gabor_Y10': [],
        'Gabor_Y11': [],
        'Gabor_L9': [],
        'Gabor_L10': [],
        'Gabor_L11': []
    }
    gabor_test = {
        'Gabor_G9': [],
        'Gabor_G10': [],
        'Gabor_G11': [],
        'Gabor_Y9': [],
        'Gabor_Y10': [],
        'Gabor_Y11': [],
        'Gabor_L9': [],
        'Gabor_L10': [],
        'Gabor_L11': []
    }
    
    print("Extracting Gabor features for train data...")
    for i in range(len(preprocessed_train['G_clahe'])):
        G_clahe = preprocessed_train['G_clahe'][i]
        Y_clahe = preprocessed_train['Y_clahe'][i]
        L_clahe = preprocessed_train['L_clahe'][i]

        gabor_features = extract_gabor_features(G_clahe, Y_clahe, L_clahe)
        gabor_train['Gabor_G9'].append(gabor_features[0])
        gabor_train['Gabor_G10'].append(gabor_features[1])
        gabor_train['Gabor_G11'].append(gabor_features[2])
        gabor_train['Gabor_Y9'].append(gabor_features[3])
        gabor_train['Gabor_Y10'].append(gabor_features[4])
        gabor_train['Gabor_Y11'].append(gabor_features[5])
        gabor_train['Gabor_L9'].append(gabor_features[6])
        gabor_train['Gabor_L10'].append(gabor_features[7])
        gabor_train['Gabor_L11'].append(gabor_features[8])

        # Save figure 5 
        save_fig5(gabor_features[0:3], gabor_features[3:6], gabor_features[6:9],
                   f"results/fig5/train_fig5_{i+21}.png")

    print("Extracting Gabor features for test data...")
    for i in range(len(preprocessed_test['G_clahe'])):
        G_clahe = preprocessed_test['G_clahe'][i]
        Y_clahe = preprocessed_test['Y_clahe'][i]
        L_clahe = preprocessed_test['L_clahe'][i]

        gabor_features = extract_gabor_features(G_clahe, Y_clahe, L_clahe)
        gabor_test['Gabor_G9'].append(gabor_features[0])
        gabor_test['Gabor_G10'].append(gabor_features[1])
        gabor_test['Gabor_G11'].append(gabor_features[2])
        gabor_test['Gabor_Y9'].append(gabor_features[3])
        gabor_test['Gabor_Y10'].append(gabor_features[4])
        gabor_test['Gabor_Y11'].append(gabor_features[5])
        gabor_test['Gabor_L9'].append(gabor_features[6])
        gabor_test['Gabor_L10'].append(gabor_features[7])
        gabor_test['Gabor_L11'].append(gabor_features[8])

        # Save figure 5 
        save_fig5(gabor_features[0:3], gabor_features[3:6], gabor_features[6:9],
                   f"results/fig5/test_fig5_{i+1:02d}.png")
    
    return gabor_train, gabor_test

def automatic_thresholding_gabor(gabor_train, gabor_test):
    """
    Apply automatic thresholding to Gabor features for entire dataset.
    """
    
    binary_train = {
        'Gabor_G9': [],
        'Gabor_G10': [],
        'Gabor_G11': [],
        'Gabor_Y9': [],
        'Gabor_Y10': [],
        'Gabor_Y11': [],
        'Gabor_L9': [],
        'Gabor_L10': [],
        'Gabor_L11': []
    }
    binary_test = {
        'Gabor_G9': [],
        'Gabor_G10': [],
        'Gabor_G11': [],
        'Gabor_Y9': [],
        'Gabor_Y10': [],
        'Gabor_Y11': [],
        'Gabor_L9': [],
        'Gabor_L10': [],
        'Gabor_L11': []
        }

    print("Applying automatic thresholding to train Gabor features...")
    for i in range(len(gabor_train['Gabor_G9'])):
        binary_train['Gabor_G9'].append(binarize_gabor_features(gabor_train['Gabor_G9'][i], preprocessed_train['cropped_mask'][i], method='otsu'))
        binary_train['Gabor_G10'].append(binarize_gabor_features(gabor_train['Gabor_G10'][i], preprocessed_train['cropped_mask'][i], method='otsu'))
        binary_train['Gabor_G11'].append(binarize_gabor_features(gabor_train['Gabor_G11'][i], preprocessed_train['cropped_mask'][i], method='otsu'))
        binary_train['Gabor_Y9'].append(binarize_gabor_features(gabor_train['Gabor_Y9'][i], preprocessed_train['cropped_mask'][i], method='otsu'))
        binary_train['Gabor_Y10'].append(binarize_gabor_features(gabor_train['Gabor_Y10'][i], preprocessed_train['cropped_mask'][i], method='otsu'))
        binary_train['Gabor_Y11'].append(binarize_gabor_features(gabor_train['Gabor_Y11'][i], preprocessed_train['cropped_mask'][i], method='otsu'))
        binary_train['Gabor_L9'].append(binarize_gabor_features(gabor_train['Gabor_L9'][i], preprocessed_train['cropped_mask'][i], method='otsu'))
        binary_train['Gabor_L10'].append(binarize_gabor_features(gabor_train['Gabor_L10'][i], preprocessed_train['cropped_mask'][i], method='otsu'))
        binary_train['Gabor_L11'].append(binarize_gabor_features(gabor_train['Gabor_L11'][i], preprocessed_train['cropped_mask'][i], method='otsu'))


        # # Save figure 6
        Binary_G = [binary_train['Gabor_G9'][i], binary_train['Gabor_G10'][i], binary_train['Gabor_G11'][i]]
        Binary_Y = [binary_train['Gabor_Y9'][i], binary_train['Gabor_Y10'][i], binary_train['Gabor_Y11'][i]]
        Binary_L = [binary_train['Gabor_L9'][i], binary_train['Gabor_L10'][i], binary_train['Gabor_L11'][i]]
        save_fig6(Binary_G, Binary_Y, Binary_L,
                   f"results/fig6/train_fig6_{i+21}.png")


    print("Applying automatic thresholding to test Gabor features...")
    for i in range(len(gabor_test['Gabor_G9'])):
        binary_test['Gabor_G9'].append(binarize_gabor_features(gabor_test['Gabor_G9'][i], preprocessed_test['cropped_mask'][i], method='otsu'))
        binary_test['Gabor_G10'].append(binarize_gabor_features(gabor_test['Gabor_G10'][i], preprocessed_test['cropped_mask'][i], method='otsu'))
        binary_test['Gabor_G11'].append(binarize_gabor_features(gabor_test['Gabor_G11'][i], preprocessed_test['cropped_mask'][i], method='otsu'))
        binary_test['Gabor_Y9'].append(binarize_gabor_features(gabor_test['Gabor_Y9'][i], preprocessed_test['cropped_mask'][i], method='otsu'))
        binary_test['Gabor_Y10'].append(binarize_gabor_features(gabor_test['Gabor_Y10'][i], preprocessed_test['cropped_mask'][i], method='otsu'))
        binary_test['Gabor_Y11'].append(binarize_gabor_features(gabor_test['Gabor_Y11'][i], preprocessed_test['cropped_mask'][i], method='otsu'))
        binary_test['Gabor_L9'].append(binarize_gabor_features(gabor_test['Gabor_L9'][i], preprocessed_test['cropped_mask'][i], method='otsu'))
        binary_test['Gabor_L10'].append(binarize_gabor_features(gabor_test['Gabor_L10'][i], preprocessed_test['cropped_mask'][i], method='otsu'))
        binary_test['Gabor_L11'].append(binarize_gabor_features(gabor_test['Gabor_L11'][i], preprocessed_test['cropped_mask'][i], method='otsu'))
    
        # # Save figure 6
        Binary_G = [binary_test['Gabor_G9'][i], binary_test['Gabor_G10'][i], binary_test['Gabor_G11'][i]]
        Binary_Y = [binary_test['Gabor_Y9'][i], binary_test['Gabor_Y10'][i], binary_test['Gabor_Y11'][i]]
        Binary_L = [binary_test['Gabor_L9'][i], binary_test['Gabor_L10'][i], binary_test['Gabor_L11'][i]]
        save_fig6(Binary_G, Binary_Y, Binary_L,
                   f"results/fig6/test_fig6_{i+1:02d}.png")

    return binary_train, binary_test

def extract_th_features(preprocessed_train, preprocessed_test):
    """Extract TH features for entire dataset."""

    print("Extracting TH features for train data...")
    th_train = []
    for i in range(len(preprocessed_train['G_clahe'])):
        G_clahe = preprocessed_train['G_clahe'][i]
        TH_feature = top_hat_extraction(G_clahe)
        th_train.append(TH_feature)

        # Save figure 7
        save_fig7(TH_feature, f"results/fig7/train_fig7_{i+21}.png")

    print("Extracting TH features for test data...")
    th_test = []
    for i in range(len(preprocessed_test['G_clahe'])):
        G_clahe = preprocessed_test['G_clahe'][i]
        TH_feature = top_hat_extraction(G_clahe)
        th_test.append(TH_feature)

        # Save figure 7
        save_fig7(TH_feature, f"results/fig7/test_fig7_{i+1:02d}.png")
    
    return th_train, th_test

def extract_sc_features(preprocessed_train, preprocessed_test):
    """Extract SC features for entire dataset."""

    print("Extracting SC features for train data...")
    sc_train = []
    for i in range(len(preprocessed_train['G_clahe'])):
        G_clahe = preprocessed_train['G_clahe'][i]
        SC_feature = SC_extraction(G_clahe)
        sc_train.append(SC_feature)

        save_fig8(SC_feature, f"results/fig8/train_fig8_{i+21}.png")

    print("Extracting SC features for test data...")
    sc_test = []
    for i in range(len(preprocessed_test['G_clahe'])):
        G_clahe = preprocessed_test['G_clahe'][i]
        SC_feature = SC_extraction(G_clahe)
        sc_test.append(SC_feature)

        save_fig8(SC_feature, f"results/fig8/test_fig8_{i+1:02d}.png")
    
    return sc_train, sc_test

def create_feature_vector_dataset(gabor_train, th_train, sc_train, gabor_test, th_test, sc_test, G_clahe_train, G_clahe_test):
    """Create feature vectors for entire dataset."""
    feature_vectors_train = []
    feature_vectors_test = []
    
    print("Creating feature vectors for train data...")
    for i in range(len(gabor_train['Gabor_G9'])):
        gabor_features = [
            gabor_train['Gabor_G9'][i],
            gabor_train['Gabor_G10'][i],
            gabor_train['Gabor_G11'][i],
            gabor_train['Gabor_Y9'][i],
            gabor_train['Gabor_Y10'][i],
            gabor_train['Gabor_Y11'][i],
            gabor_train['Gabor_L9'][i],
            gabor_train['Gabor_L10'][i],
            gabor_train['Gabor_L11'][i]
        ]
        th_feature = th_train[i]
        sc_feature = sc_train[i]
        G_clahe = G_clahe_train[i]
        feature_vector = create_feature_vector(gabor_features, G_clahe, th_feature, sc_feature)
        feature_vectors_train.append(feature_vector)
    
    print("Creating feature vectors for test data...")
    for i in range(len(gabor_test['Gabor_G9'])):
        gabor_features = [
            gabor_test['Gabor_G9'][i],
            gabor_test['Gabor_G10'][i],
            gabor_test['Gabor_G11'][i],
            gabor_test['Gabor_Y9'][i],
            gabor_test['Gabor_Y10'][i],
            gabor_test['Gabor_Y11'][i],
            gabor_test['Gabor_L9'][i],
            gabor_test['Gabor_L10'][i],
            gabor_test['Gabor_L11'][i]
        ]
        th_feature = th_test[i]
        sc_feature = sc_test[i]
        G_clahe = G_clahe_test[i]
        feature_vector = create_feature_vector(gabor_features, G_clahe, th_feature, sc_feature)
        feature_vectors_test.append(feature_vector)
    
    return feature_vectors_train, feature_vectors_test

def apply_pca(feature_vectors_train, feature_vectors_test):
    """Apply PCA to feature vectors."""
    print("Applying PCA...")

    # Stack per-image feature vectors into big arrays, remembering sizes to split back later
    train_sizes = [fv.shape[0] for fv in feature_vectors_train]
    test_sizes = [fv.shape[0] for fv in feature_vectors_test]

    X_train = np.vstack(feature_vectors_train)
    X_test = np.vstack(feature_vectors_test)

    scaler = StandardScaler()
    scaler.fit(X_train)

    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    pca = PCA(n_components=12)
    pca.fit(X_train_scaled)

    X_train_pca_all = pca.transform(X_train_scaled)
    X_test_pca_all = pca.transform(X_test_scaled)

    # Split back into per-image arrays
    train_split_idx = np.cumsum(train_sizes)[:-1]
    test_split_idx = np.cumsum(test_sizes)[:-1]

    X_train_pca = list(np.split(X_train_pca_all, train_split_idx, axis=0)) 
    X_test_pca = list(np.split(X_test_pca_all, test_split_idx, axis=0)) 
    
    return X_train_pca, X_test_pca

def apply_fcm(X_train_pca, X_test_pca, n_clusters=2, m=2.0, error=1e-6, maxiter=2000, seed=40):
    """
    Apply FCM clustering to PCA-reduced features.
    Returns list of binary predictions for train and test sets.
    """
    clustered_train = []
    clustered_test = []

    print("Applying FCM to train data...")
    for i in range(len(X_train_pca)):
        labels, U, centers, fpc = fcm_fit_predict(X_train_pca[i], n_clusters, m, error, maxiter, seed)
        vessel_cluster = find_cluster_with_less_pixels(labels)
        binary_prediction = binary_fcm_prediction(labels, vessel_cluster)
        clustered_train.append(binary_prediction)

        #save figure 10
        save_fig10(binary_prediction.reshape(preprocessed_train['cropped_mask'][i].shape), f"results/fig10/train_fig10_{i+21}.png")
    
    print("Applying FCM to test data...")
    for i in range(len(X_test_pca)):
        labels, U, centers, fpc = fcm_fit_predict(X_test_pca[i], n_clusters, m, error, maxiter, seed)
        vessel_cluster = find_cluster_with_less_pixels(labels)
        binary_prediction = binary_fcm_prediction(labels, vessel_cluster)
        clustered_test.append(binary_prediction)

        #save figure 10
        save_fig10(binary_prediction.reshape(preprocessed_test['cropped_mask'][i].shape), f"results/fig10/test_fig10_{i+1:02d}.png")

    return clustered_train, clustered_test

def apply_fast_classifier_on_nonvessel_pixels(
    X_train_pca, X_test_pca,
    binary_labels_train, binary_labels_test,
    preprocessed_train, preprocessed_test,
    train_gt,
    n_estimators=300,
    samples_per_train_image=4000,
    prob_threshold=0.5,
    random_state=42
):
    """
    Apply FAST classifier (ExtraTrees) to non-vessel pixels to detect missed vessels.
    Parameters:
    - X_train_pca, X_test_pca: PCA-reduced feature vectors for train and test images.
    - binary_labels_train, binary_labels_test: Binary labels from FCM clustering (1 for vessel, 0 for non-vessel).
    - preprocessed_train, preprocessed_test: Preprocessed data tuples for train and test images.
    - train_gt: Ground truth vessel masks for training images (used for labeling non-vessel pixels).
    - n_estimators: Number of trees in the ExtraTrees ensemble.
    - samples_per_train_image: Maximum number of non-vessel pixels to sample from each training image for training the classifier.
    - prob_threshold: Probability threshold for classifying a pixel as vessel in the test set.
    - random_state: Random seed for reproducibility.
    Returns:
    - refined_train: List of refined binary vessel masks for training images after applying the classifier.
    - refined_test: List of refined binary vessel masks for test images after applying the classifier.
    - prob_maps: List of probability maps for test images indicating the likelihood of each pixel being a vessel according to the classifier.
    """

    print("Applying FAST classifier (ExtraTrees) on non-vessel pixels...")

    rng = np.random.default_rng(random_state)

    X_all = []
    y_all = []

    for i in range(len(X_train_pca)):
        cropped_mask = preprocessed_train['cropped_mask'][i]  # FOV crop mask
        bounds = preprocessed_train['bounds'][i]
        y_min, y_max, x_min, x_max = bounds
        gt_crop = train_gt[i][y_min:y_max+1, x_min:x_max+1]

        gt_flat = (gt_crop.reshape(-1) > 0).astype(np.uint8)
        fov_flat = (cropped_mask.reshape(-1) > 0)

        nonv_flat = (binary_labels_train[i].reshape(-1) == 0)
        cand = np.where(fov_flat & nonv_flat)[0]
        if cand.size == 0:
            continue

        take = min(samples_per_train_image, cand.size)
        sel = rng.choice(cand, size=take, replace=False)

        X_all.append(X_train_pca[i][sel].astype(np.float32))
        y_all.append(gt_flat[sel])

    X_all = np.vstack(X_all)
    y_all = np.concatenate(y_all)

    clf = ExtraTreesClassifier(
        n_estimators=n_estimators,
        max_depth=None,
        min_samples_leaf=2,
        max_features="sqrt",
        n_jobs=-1,
        random_state=random_state,
        class_weight="balanced_subsample",
    )
    clf.fit(X_all, y_all)

    if 1 in clf.classes_:
        c1 = np.where(clf.classes_ == 1)[0][0]
    else:
        c1 = None

    #Apply to train
    refined_train = []
    for i in range(len(X_train_pca)):
        cropped_mask = preprocessed_train['cropped_mask'][i]
        H, W = cropped_mask.shape

        vessel_from_cluster = (binary_labels_train[i].reshape(-1) == 1)
        refined_flat = vessel_from_cluster.copy()

        if c1 is not None:
            nonv = (binary_labels_train[i].reshape(-1) == 0)
            fov = (cropped_mask.reshape(-1) > 0)
            idx = np.where(nonv & fov)[0]

            if idx.size > 0:
                proba = clf.predict_proba(X_train_pca[i][idx].astype(np.float32))
                refined_flat[idx] |= (proba[:, c1] >= prob_threshold)

        image = refined_flat.reshape(H, W).astype(np.uint8)
        save_fig11(image, f"results/fig11/train_fig11_{i+21}.png")

        refined_train.append(image)


    #Apply to test
    refined_test = []
    prob_maps = []
    for i in range(len(X_test_pca)):
        cropped_mask = preprocessed_test['cropped_mask'][i]
        H, W = cropped_mask.shape

        vessel_from_cluster = (binary_labels_test[i].reshape(-1) == 1)
        refined_flat = vessel_from_cluster.copy()
        prob_flat = np.zeros(H * W, dtype=float)

        if c1 is not None:
            nonv = (binary_labels_test[i].reshape(-1) == 0)
            fov = (cropped_mask.reshape(-1) > 0)
            idx = np.where(nonv & fov)[0]

            if idx.size > 0:
                proba = clf.predict_proba(X_test_pca[i][idx].astype(np.float32))
                prob_flat[idx] = proba[:, c1]
                refined_flat[idx] |= (proba[:, c1] >= prob_threshold)

        image = refined_flat.reshape(H, W).astype(np.uint8)
        save_fig11(image, f"results/fig11/test_fig11_{i+1:02d}.png")
        refined_test.append(image)
        prob_maps.append(prob_flat.reshape(H, W))

    return refined_train, refined_test, prob_maps

def combine_results(clustered_test, refined_test, preprocessed_test, clustered_train, refined_train, preprocessed_train):
    """Combine FCM and FAST results by taking pixel-wise OR."""
    
    combined_train = []
    for i in range(len(clustered_train)):
        clustered_i = clustered_train[i].reshape(preprocessed_train['cropped_mask'][i].shape).astype(bool)
        refined_i = refined_train[i].astype(bool)
        combined = (clustered_i | refined_i).astype(np.uint8) * 255
        combined_train.append(combined)
    
    combined_test = []
    for i in range(len(clustered_test)):
        clustered_i = clustered_test[i].reshape(preprocessed_test['cropped_mask'][i].shape).astype(bool)
        refined_i = refined_test[i].astype(bool)
        combined = (clustered_i | refined_i).astype(np.uint8) * 255
        combined_test.append(combined)

    return combined_test, combined_train

def apply_post_processing(refined_train, refined_test, preprocessed_train, preprocessed_test):
    """Apply post-processing to remove FOV border ring and small objects."""
    final_train = []
    final_test = []

    print("Applying post-processing to train data...")
    for i in range(len(refined_train)):
        cleaned = remove_fov_ring_only(refined_train[i], preprocessed_train[i], border_px=12, outside_zero=True)
        save_fig12(cleaned, f"results/fig12/train_fig12_{i+21}.png")
        final_train.append(cleaned)

    print("Applying post-processing to test data...")
    for i in range(len(refined_test)):
        cleaned = remove_fov_ring_only(refined_test[i], preprocessed_test[i], border_px=12, outside_zero=True)
        save_fig12(cleaned, f"results/fig12/test_fig12_{i+1:02d}.png")
        final_test.append(cleaned)

    return final_train, final_test

def evaluate_results(final_test, test_gt, preprocessed_test):

    #step1 crop GT to FOV and flatten
    cropped_gt_test = []
    for i in range(len(test_gt)):
        y_min, y_max, x_min, x_max = preprocessed_test['bounds'][i]
        gt_crop = test_gt[i][y_min:y_max+1, x_min:x_max+1]
        cropped_gt_test.append(gt_crop.reshape(-1) > 0)

    #step2 calculate tp, fp, fn, tn for each image and average metrics 
    accuracy_scores = []
    sensitivities = []
    specificities = []
    precision_scores = []

    for i in range(len(final_test)):
        pred_flat = (final_test[i].reshape(-1) > 0)
        mask_flat = (preprocessed_test['cropped_mask'][i].reshape(-1) > 0)
        gt_flat = cropped_gt_test[i]

        tp = np.sum(pred_flat & gt_flat & mask_flat)
        fp = np.sum(pred_flat & ~gt_flat & mask_flat)
        fn = np.sum(~pred_flat & gt_flat & mask_flat)
        tn = np.sum(~pred_flat & ~gt_flat & mask_flat)

        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 1.0
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 1.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 1.0
        precision_value = tp / (tp + fp) if (tp + fp) > 0 else 1.0

        accuracy_scores.append(accuracy)
        sensitivities.append(sensitivity)
        specificities.append(specificity)
        precision_scores.append(precision_value)
    
    Avage_Accuracy = np.mean(accuracy_scores)
    Avage_Sensitivity = np.mean(sensitivities)
    Avage_Specificity = np.mean(specificities)
    Avage_Precision = np.mean(precision_scores)

    #save average metrics as csv 
    with open("results/metrics.csv", mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Metric", "Value"])
        writer.writerow(["Average Accuracy", Avage_Accuracy])
        writer.writerow(["Average Sensitivity", Avage_Sensitivity])
        writer.writerow(["Average Specificity", Avage_Specificity])
        writer.writerow(["Average Precision", Avage_Precision])

    return Avage_Accuracy, Avage_Sensitivity, Avage_Specificity, Avage_Precision
# ========================= SAVING FIGURES ==========================

def save_fig3(G_channel, Y_channel, L_channel, save_path): 
    """Save figure with G, Y, L channels in grayscale."""
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.imshow(G_channel, cmap='gray')
    plt.title('G Channel')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(Y_channel, cmap='gray')
    plt.title('Y Channel')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(L_channel, cmap='gray')
    plt.title('L Channel')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def save_fig4(G_clahe, Y_clahe, L_clahe, save_path):
    """Save figure with CLAHE enhanced G, Y, L channels in grayscale."""
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.imshow(G_clahe, cmap='gray')
    plt.title('G Channel CLAHE')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(Y_clahe, cmap='gray')
    plt.title('Y Channel CLAHE')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(L_clahe, cmap='gray')
    plt.title('L Channel CLAHE')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def save_fig5(Gabor_G, Gabor_Y, Gabor_L, save_path):
    """Save figure with Gabor filter responses for G, Y, L channels for 3 wavelengths."""
    plt.figure(figsize=(12, 8))
    
    wavelengths = [9, 10, 11]
    
    for i, wavelength in enumerate(wavelengths):
        plt.subplot(3, 3, i + 1)
        plt.imshow(Gabor_G[i], cmap='gray')
        plt.title(f'Gabor G - Wavelength {wavelength}')
        plt.axis('off')
        
        plt.subplot(3, 3, i + 4)
        plt.imshow(Gabor_Y[i], cmap='gray')
        plt.title(f'Gabor Y - Wavelength {wavelength}')
        plt.axis('off')
        
        plt.subplot(3, 3, i + 7)
        plt.imshow(Gabor_L[i], cmap='gray')
        plt.title(f'Gabor L - Wavelength {wavelength}')
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
   
def save_fig6(binary_G, binary_Y, binary_L, save_path):
    """Save figure with Gabor filter responses for G, Y, L channels for 3 wavelengths after automatic thresholding."""
    plt.figure(figsize=(12, 8))
    
    wavelengths = [9, 10, 11]
    
    for i, wavelength in enumerate(wavelengths):
        plt.subplot(3, 3, i + 1)
        plt.imshow(binary_G[i], cmap='gray')
        plt.title(f'Gabor G - Wavelength {wavelength}')
        plt.axis('off')
        
        plt.subplot(3, 3, i + 4)
        plt.imshow(binary_Y[i], cmap='gray')
        plt.title(f'Gabor Y - Wavelength {wavelength}')
        plt.axis('off')
        
        plt.subplot(3, 3, i + 7)
        plt.imshow(binary_L[i], cmap='gray')
        plt.title(f'Gabor L - Wavelength {wavelength}')
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def save_fig7(TH_feature, save_path):
    """Save figure of TH Feature."""
    plt.figure(figsize=(6, 6))
    plt.imshow(TH_feature, cmap='gray')
    plt.title('TH Feature')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def save_fig8(SC_feature, save_path):
    """Save figure of SC Feature."""
    plt.figure(figsize=(6, 6))
    plt.imshow(SC_feature, cmap='gray')
    plt.title('SC Feature')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def save_fig10(clustered_image, save_path):
    """Save figure of clustered image."""
    plt.figure(figsize=(6, 6))
    plt.imshow(clustered_image, cmap='gray')
    plt.title('Clustered Image')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def save_fig11(classified_image, save_path):
    """Save figure of classified image."""
    plt.figure(figsize=(6, 6))
    plt.imshow(classified_image, cmap='gray')
    plt.title('Classified Image')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def save_fig12(final_image, save_path):
    """Save figure of final post-processed image."""
    plt.figure(figsize=(6, 6))
    plt.imshow(final_image, cmap='gray')
    plt.title('Final Post-Processed Image')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# ========================== MAIN START ==========================

"""
Load train and test images, masks, and ground truths from DRIVE dataset.
Return lists of images, masks, and ground truths for both train and test sets.
"""
data_path = "DRIVE"  # Update this path as needed
train_images, train_masks, train_gt, test_images, test_masks, test_gt = load_drive_dataset(data_path)

"""
Apply preprocessing pipeline to each image in the dataset. 
Returns a dictionary for each image containing cropped image, cropped mask, G/Y/L channels before and after CLAHE, and crop bounds.
"""
preprocessed_train, preprocessed_test = preprocess_dataset(train_images, train_masks, test_images, test_masks)

"""
Extract Gabor features for each image in the dataset.
For each image, save list of 9 Gabor features (3 channels × 3 wavelengths) in a list.
"""
gray_gabor_train , gray_gabor_test = gabor_feature_extraction_dataset(preprocessed_train, preprocessed_test)

""" 
Binarize Gabor features using Laplacian enhancement + automatic thresholding.
For each image, save list of 9 binary Gabor features in a list.
"""
binary_gabor_train, binary_gabor_test = automatic_thresholding_gabor(gray_gabor_train, gray_gabor_test)

""" 
Extract TH features for each image in the dataset.
For each image, save TH feature in a list.
"""
th_train, th_test = extract_th_features(preprocessed_train, preprocessed_test)

""" 
Extract SC features for each image in the dataset.
For each image, save SC feature in a list.
"""
sc_train, sc_test = extract_sc_features(preprocessed_train, preprocessed_test)

""" 
Create feature vectors for each image in the dataset.
For each image, save feature vector of shape (num_pixels, num_features) in a list.
"""
feature_vectors_train, feature_vectors_test = create_feature_vector_dataset(
    binary_gabor_train,
    th_train,
    sc_train,
    binary_gabor_test, 
    th_test, 
    sc_test, 
    preprocessed_train['G_clahe'], 
    preprocessed_test['G_clahe'])
""" 
Apply PCA for dimensionality reduction.
For each image, save reduced feature vector in a list.
Shape of each feature vector should be (num_pixels, 12) after PCA.
"""
X_train_pca, X_test_pca = apply_pca(feature_vectors_train, feature_vectors_test)

""" 
Apply Fuzzy C-Means clustering to PCA-reduced feature vectors.
For each image, save binary cluster labels (vessel=1, non-vessel=0) in a list.
"""
clustered_train, clustered_test = apply_fcm(X_train_pca, X_test_pca, n_clusters=2, m=2.0, error=1e-6, maxiter=2000, seed=40)

"""
Apply Decision Tree classifier to Non-vessel pixels to detect missed vessels.
"""
refined_train, refined_test, prob_maps = apply_fast_classifier_on_nonvessel_pixels(
    X_train_pca, X_test_pca,
    clustered_train, clustered_test,
    preprocessed_train, preprocessed_test,
    train_gt,
    n_estimators=300,
    samples_per_train_image=4000,
    prob_threshold=0.845,
    random_state=42
)

"""
Apply pixel-wise OR to combine FCM and FAST results for final vessel segmentation masks.
"""
combined_test, combined_train = combine_results(clustered_test, refined_test, preprocessed_test, clustered_train, refined_train, preprocessed_train)

"""
Apply post-processing to refine vessel segmentation results.
"""
final_train, final_test = apply_post_processing(combined_train, combined_test, preprocessed_train['cropped_mask'], preprocessed_test['cropped_mask'])

"""
Evaluate final results against ground truth masks using Accuracy, Sensitivity, Specificity, and Precision metrics.
"""
average_accuracy, average_sensitivity, average_specificity, average_precision = evaluate_results(final_test, test_gt, preprocessed_test)