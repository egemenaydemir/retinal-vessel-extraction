import cv2
import numpy as np
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_curve, auc
import os
import warnings
import matplotlib
matplotlib.use('Agg')
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

# ========================== MAIN START ==========================

def main():

    #load dataset
    data_path = "DRIVE"  # Update this path as needed
    train_images, train_masks, train_gt, test_images, test_masks, test_gt = load_drive_dataset(data_path)
    
    #apply preprocessing 
    preprocessed_train, preprocessed_test = preprocess_dataset(train_images, train_masks, test_images, test_masks)

    #apply gabor filter
    gabor_train , gabor_test = gabor_feature_extraction_dataset(preprocessed_train, preprocessed_test)

    #apply automatic thresholding to gabor images
    binary_train, binary_test = automatic_thresholding_gabor(gabor_train, gabor_test)










if __name__ == "__main__":
    main()