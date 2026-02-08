import cv2
import numpy as np 
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import StandardScaler
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
    cropped_img, bounds = crop_to_fov(image, mask)
    cropped_mask = crop_mask(mask, bounds)
    G_raw, Y_raw, L_raw = extract_channels(cropped_img)
    G_clahe = apply_clahe(G_raw)
    Y_clahe = apply_clahe(Y_raw)
    L_clahe = apply_clahe(L_raw)
    return cropped_img, cropped_mask, G_raw, Y_raw, L_raw, G_clahe, Y_clahe, L_clahe, bounds

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

def binarize_gabor_features(gabor_features):
    """
    Binarize Gabor features.
    
    Laplacian enhancement + automatic threshold (paper method)
    """
    binary_features = []
    
    for gabor in gabor_features:

        gabor_norm = cv2.normalize(gabor, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        laplacian = cv2.Laplacian(gabor_norm, cv2.CV_64F)
        laplacian = laplacian.astype(np.uint8)
        combined = (laplacian + gabor_norm).astype(np.uint8)
        
        # Calculate automatic threshold on combined image (using mask)
        threshold = calculate_automatic_threshold(combined)
        _, binary = cv2.threshold(combined, int(threshold), 255, cv2.THRESH_BINARY)
        
        binary_features.append(binary)
    
    return binary_features
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
def fcm_clustering(X, n_clusters, m, max_iter, tol, seed):
    """
    Fuzzy C-Means clustering.

    Parameters:
    - X: Input data (num_samples, num_features)
    - n_clusters: Number of clusters
    - m: Fuzziness parameter
    - max_iter: Maximum number of iterations
    - tol: Tolerance for convergence
    - seed: Random seed for reproducibility

    Returns:
    - centers: Cluster centers (n_clusters, num_features)
    - membership: Membership matrix (num_samples, n_clusters)

    """

    if m <= 1:
        raise ValueError("Fuzziness parameter m must be greater than 1.")
    

    rng = np.random.default_rng(seed)
    num_samples, num_features = X.shape

    # Initialize memberships randomly 
    membership = rng.random((num_samples, n_clusters))
    membership = membership / membership.sum(axis=1, keepdims=True)

    eps = 1e-6  # Small constant to prevent division by zero

    for _ in range(max_iter):
        old_membership = membership.copy()

        #compute cluster centers : v_k = sum_i (u_ik^m x_i) / sum_i (u_ik^m)
        centers = (membership**m).T @ X / np.sum(membership**m, axis=0)[:, np.newaxis]

        #distaces : d_ik = ||x_i - v_k||
        distances = np.linalg.norm(X[:, np.newaxis] - centers, axis=2)
        distances = np.maximum(distances, eps)

        #update memberships : u_ik = 1 / sum_j (d_ik / d_ij)^(2/(m-1))
        power = 2 / (m - 1)
        ratio = (distances[:, :, np.newaxis] / distances[:, np.newaxis, :]) ** power
        membership = 1 / np.sum(ratio, axis=2)

        #check convergence
        if np.max(np.abs(membership - old_membership)) <  tol:
            break

        labels = np.argmax(membership, axis=1)
    return centers, membership, labels

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
    preprocessed_train = []
    preprocessed_test = []
    
    print("Preprocessing train data...")
    for idx, (img, mask) in enumerate(zip(training_images, training_masks)):
        (cropped_img, cropped_mask, G_raw, Y_raw, L_raw,
         G_clahe, Y_clahe, L_clahe, bounds) = preprocess_image(img, mask)
        
        # Save figures
        save_fig3(G_raw, Y_raw, L_raw, f"results/fig3/train_fig3_{idx+21}.png")
        save_fig4(G_clahe, Y_clahe, L_clahe, f"results/fig4/train_fig4_{idx+21}.png")
        
        preprocessed_train.append((cropped_img, cropped_mask, G_raw, Y_raw, L_raw,
                                   G_clahe, Y_clahe, L_clahe, bounds))
    
    print("Preprocessing test data...")
    for idx, (img, mask) in enumerate(zip(test_images, test_masks)):
        (cropped_img, cropped_mask, G_raw, Y_raw, L_raw,
         G_clahe, Y_clahe, L_clahe, bounds) = preprocess_image(img, mask)
        
        # Save figures
        save_fig3(G_raw, Y_raw, L_raw, f"results/fig3/test_fig3_{idx+1:02d}.png")
        save_fig4(G_clahe, Y_clahe, L_clahe, f"results/fig4/test_fig4_{idx+1:02d}.png")
        
        preprocessed_test.append((cropped_img, cropped_mask, G_raw, Y_raw, L_raw,
                                  G_clahe, Y_clahe, L_clahe, bounds))
    
    return preprocessed_train, preprocessed_test
    
def gabor_feature_extraction_dataset(preprocessed_train, preprocessed_test):
    """
    Extract Gabor features for entire dataset.
    """
    gabor_train = []
    gabor_test = []
    
    print("Extracting Gabor features for train data...")
    for idx, (_, _, _, _, _, G_clahe, Y_clahe, L_clahe, _) in enumerate(preprocessed_train):
        
        gabor_features = extract_gabor_features(G_clahe, Y_clahe, L_clahe)
        gabor_train.append(gabor_features)

        # Save figure 5 
        save_fig5(gabor_features[0:3], gabor_features[3:6], gabor_features[6:9],
                   f"results/fig5/train_fig5_{idx+21}.png")

    
    print("Extracting Gabor features for test data...")
    for idx, (_, _, _, _, _, G_clahe, Y_clahe, L_clahe, _) in enumerate(preprocessed_test):
        
        gabor_features = extract_gabor_features(G_clahe, Y_clahe, L_clahe)
        gabor_test.append(gabor_features)

        # Save figure 5 
        save_fig5(gabor_features[0:3], gabor_features[3:6], gabor_features[6:9],
                   f"results/fig5/test_fig5_{idx+1:02d}.png")
    
    return gabor_train, gabor_test

def automatic_thresholding_gabor(gabor_train, gabor_test):
    """
    Apply automatic thresholding to Gabor features for entire dataset.
    """
    
    binary_train = []
    binary_test = []

    print("Applying automatic thresholding to train Gabor features...")
    for idx, gabor_features in enumerate(gabor_train):
        binary_features = binarize_gabor_features(gabor_features)
        binary_train.append(binary_features)

        # Save figure 6
        save_fig6(binary_features[0:3], binary_features[3:6], binary_features[6:9],
                   f"results/fig6/train_fig6_{idx+21}.png")


    print("Applying automatic thresholding to test Gabor features...")
    for idx, gabor_features in enumerate(gabor_test):
        binary_features = binarize_gabor_features(gabor_features)
        binary_test.append(binary_features)

        # Save figure 6
        save_fig6(binary_features[0:3], binary_features[3:6], binary_features[6:9],
                   f"results/fig6/test_fig6_{idx+1:02d}.png")

    return binary_train, binary_test

def extract_th_features(preprocessed_train, preprocessed_test):
    """Extract TH features for entire dataset."""

    print("Extracting TH features for train data...")
    th_train = []
    for idx, (_, _, _, _, _, G_clahe, _, _, _) in enumerate(preprocessed_train):
        TH_feature = top_hat_extraction(G_clahe)
        th_train.append(TH_feature)

        save_fig7(TH_feature, f"results/fig7/train_fig7_{idx+21}.png")

    print("Extracting TH features for test data...")
    th_test = []
    for idx, (_, _, _, _, _, G_clahe, _, _, _) in enumerate(preprocessed_test):
        TH_feature = top_hat_extraction(G_clahe)
        th_test.append(TH_feature)

        save_fig7(TH_feature, f"results/fig7/test_fig7_{idx+1:02d}.png")
    
    return th_train, th_test

def extract_sc_features(preprocessed_train, preprocessed_test):
    """Extract SC features for entire dataset."""

    print("Extracting SC features for train data...")
    sc_train = []
    for idx, (_, _, _, _, _, G_clahe, _, _, _) in enumerate(preprocessed_train):
        SC_feature = SC_extraction(G_clahe)
        sc_train.append(SC_feature)

        save_fig8(SC_feature, f"results/fig8/train_fig8_{idx+21}.png")

    print("Extracting SC features for test data...")
    sc_test = []
    for idx, (_, _, _, _, _, G_clahe, _, _, _) in enumerate(preprocessed_test):
        SC_feature = SC_extraction(G_clahe)
        sc_test.append(SC_feature)

        save_fig8(SC_feature, f"results/fig8/test_fig8_{idx+1:02d}.png")
    
    return sc_train, sc_test

def create_feature_vector_dataset(gabor_train, th_train, sc_train, gabor_test, th_test, sc_test, G_clahe_train, G_clahe_test):
    """Create feature vectors for entire dataset."""
    feature_vectors_train = []
    feature_vectors_test = []
    
    print("Creating feature vectors for train data...")
    for gabor_features, th_feature, sc_feature, G_clahe in zip(gabor_train, th_train, sc_train, G_clahe_train):
        feature_vector = create_feature_vector(gabor_features, G_clahe, th_feature, sc_feature)
        feature_vectors_train.append(feature_vector)
    
    print("Creating feature vectors for test data...")
    for gabor_features, th_feature, sc_feature, G_clahe in zip(gabor_test, th_test, sc_test, G_clahe_test):
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

    # Scale using training statistics, then transform test with same scaler
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Apply PCA (keep 95% variance)
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

def apply_fcm_clustering(X_train_pca, X_test_pca, train_masks, test_masks, n_clusters=2, m=2, max_iter=2000, tol=1e-6, seed=42):
    """Apply Fuzzy C-Means clustering to PCA-reduced feature vectors."""

    print("Applying Fuzzy C-Means clustering...")

    binary_labels_train = []
    for idx, X in enumerate(X_train_pca):
        centers, membership, labels = fcm_clustering(X, n_clusters, m, max_iter, tol, seed)
        binary_labels = (labels == 1).astype(np.uint8) 
        
        #number of pixels in each cluster
        cluster_counts = np.bincount(labels)
        vessel_cluster = np.argmin(cluster_counts)        #cluster with less pixel is vessel cluster
        
        #assign vessel cluster to 1 and non-vessel cluster to 0
        binary_labels = (labels == vessel_cluster).astype(np.uint8)
        binary_labels_train.append(binary_labels)

        #reshape back to original image shape
        binary_image = binary_labels.reshape(train_masks[idx].shape)  # Reshape to original image shape
        save_fig10(binary_image, f"results/fig10/train_fig10_{idx+21}.png")

    binary_labels_test = []
    for idx, X in enumerate(X_test_pca):
        centers, membership, labels = fcm_clustering(X, n_clusters, m, max_iter, tol, seed)
        binary_labels = (labels == 1).astype(np.uint8)  

        #number of pixels in each cluster
        cluster_counts = np.bincount(labels)
        vessel_cluster = np.argmin(cluster_counts)        #cluster with less pixel is vessel cluster

        #assign vessel cluster to 1 and non-vessel cluster to 0
        binary_labels = (labels == vessel_cluster).astype(np.uint8)
        binary_labels_test.append(binary_labels)

        #reshape back to original image shape
        binary_image = binary_labels.reshape(test_masks[idx].shape)  # Reshape to original image shape
        save_fig10(binary_image, f"results/fig10/test_fig10_{idx+1:02d}.png")

    return binary_labels_train, binary_labels_test

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

# ========================== MAIN START ==========================

"""
Load train and test images, masks, and ground truths from DRIVE dataset.
Return lists of images, masks, and ground truths for both train and test sets.
"""
data_path = "DRIVE"  # Update this path as needed
train_images, train_masks, train_gt, test_images, test_masks, test_gt = load_drive_dataset(data_path)

"""
Apply preprocessing pipeline to each image in the dataset. 
Shape of each item in the tuple should be:
(0)cropped image: (H_crop, W_crop, 3)
(1)cropped mask: (H_crop, W_crop)
(2)G raw: (H_crop, W_crop)
(3)Y raw: (H_crop, W_crop)
(4)L raw: (H_crop, W_crop)
(5)G CLAHE: (H_crop, W_crop)
(6)Y CLAHE: (H_crop, W_crop)
(7)L CLAHE: (H_crop, W_crop)
(8)bounds: (y_min, y_max, x_min, x_max)
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
gabor_train, gabor_test = automatic_thresholding_gabor(gray_gabor_train, gray_gabor_test)

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
g_clahe_train = [item[5] for item in preprocessed_train]
g_clahe_test = [item[5] for item in preprocessed_test]
feature_vectors_train, feature_vectors_test = create_feature_vector_dataset(gabor_train, th_train, sc_train,gabor_test, th_test, sc_test,g_clahe_train, g_clahe_test)

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
train_masks = [item[1] for item in preprocessed_train]
test_masks = [item[1] for item in preprocessed_test]
binary_labels_train, binary_labels_test = apply_fcm_clustering(X_train_pca, X_test_pca, train_masks, test_masks)