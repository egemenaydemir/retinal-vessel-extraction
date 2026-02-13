# Retinal Blood Vessel Extraction: A Hybrid ML Approach

This project was developed as the **Final Project for the Computer Vision course** at **Izmir Institute of Technology (IZTECH)**. 

It is an implementation of the methodology proposed by **Mahdi Hashemzadeh** and **Baharak Adlpour Azar** in their 2019 research paper: *"Retinal blood vessel extraction employing effective image features and combination of supervised and unsupervised machine learning methods"*.

## 📌 Project Overview
Retinal vessel analysis is essential for diagnosing various ophthalmological and cardiovascular diseases. This project implements a hybrid pipeline that combines the speed of unsupervised clustering (FCM) with the precision of supervised classification to extract complex vascular structures from fundus images.

This implementation is designed to process the DRIVE dataset , which contains a total of 40 retinal images divided into 20 training images and 20 test images. For each of these 40 images, the code executes the full pipeline and saves visual outputs at every critical stage to the results/ folder, ensuring complete transparency of the algorithm's performance.

## 📂 File Structure
The repository is organized to maintain a clear distinction between the input dataset and the generated pipeline outputs:

```text
├── main.py                 # Core implementation script
├── DRIVE/                  # Root dataset folder 
│   ├── training/           # 20 images for feature extraction and training
│   │   ├── images/         # Original fundus images 
│   │   ├── mask/           # Binary Field of View (FOV) masks 
│   │   └── 1st_manual/     # Expert ground truth vessel masks 
│   └── test/               # 20 images for final evaluation 
│       ├── images/
│       ├── mask/
│       └── 1st_manual/
└── results/                # Generated outputs for all 40 images 
    ├── fig3/               # Color channel extraction (G, Y, L) 
    ├── fig4/               # CLAHE contrast enhancement 
    ├── fig5/               # Gabor filter responses 
    ├── fig6/               # Automatic thresholding results 
    ├── fig7/               # Top-Hat (TH) features 
    ├── fig8/               # Shade Corrected (SC) features 
    ├── fig10/              # FCM Clustering results 
    ├── fig11/              # Classifier results
    ├── fig12/              # Final post-processed segmented images 
    └── metrics.csv         # Performance scores (Acc, Sen, Spe, Precision) 
```

## 🛠 Methodology
The system follows the modular structure defined in the original research:

### 1. Pre-processing
* **FOV Extraction:** Cropping to the Field of View to focus on relevant pixels.
* **Color Space Transformation:** Converting RGB to **YCbCr** and **L*a*b** to extract the **G, Y, and L channels**.
* **Contrast Enhancement:** Applying **CLAHE** to improve vessel discriminability.

### 2. Vessel Extraction
* **Feature Extraction:** Constructing a feature vector for each pixel using:
    * **Gabor Features:** 9 multi-scale Gabor filter responses.
    * **Top-Hat (TH) Transform:** Highlights linear vessel structures.
    * **Shade Corrected (SC):** Eliminates background light intensity variations.
* **PCA:** Reducing feature redundancy and eliminating correlations.
* **Hybrid Step:**
    * **Unsupervised (FCM):** Fuzzy C-Means clustering identifies thick and clear vessels.
    * **Supervised Classifier:** Targets thin and faint vessels missed in the clustering phase.

## 🔄 Implementation Differences
To adapt to the implementation environment, the following changes were made from the original paper:

* **Feature Selection:** The **Bit Plane Slicing (BPS)** feature mentioned in the paper was **not used** in this implementation to maintain focus on the most effective spatial features.
* **Supervised Model:** Instead of the Root Guided Decision Tree, an **Extra Trees Classifier** was utilized. This provided better generalization for detecting thin vessels in non-vessel regions.
* **Gabor Features Thresholding Strategy:** Both the paper's Laplacian-based automatic thresholding and **Otsu's Method** were implemented. In this specific implementation, **Otsu’s method** yielded superior segmentation results for Gabor features and was selected for the final results.

## 📊 Reference Performance (Original Research)
| Database | Accuracy ($Acc$) | Sensitivity ($Sen$) | Specificity ($Spe$) | AUC |
| :--- | :--- | :--- | :--- | :--- |
| **DRIVE** | 0.9531 | 0.7830 | 0.9800 | 0.9752 |

## 💻 Technical Requirements
* **Language:** Python 3.x
* **Core Libraries:** `OpenCV`, `NumPy`, `Scikit-learn`, `Scikit-fuzzy`, `Matplotlib`.

## 📖 Citation
Original paper:
```bibtex
@article{hashemzadeh2019retinal,
  title={Retinal blood vessel extraction employing effective image features and combination of supervised and unsupervised machine learning methods},
  author={Hashemzadeh, Mahdi and Adlpour Azar, Baharak},
  journal={Artificial Intelligence in Medicine},
  volume={95},
  pages={1--15},
  year={2019},
  publisher={Elsevier}
}
