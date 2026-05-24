import numpy as np
import cv2
from skimage.feature import hog, local_binary_pattern

def extract_hog(img):
    return hog(img, orientations=8, pixels_per_cell=(7, 7), cells_per_block=(2, 2), visualize=False)

def extract_lbp(img):
    lbp = local_binary_pattern(img, P=8, R=1, method='uniform')
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 12), range=(0, 11))
    return hist.astype("float32") / (hist.sum() + 1e-6)

def extract_edges(img):
    sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    return cv2.resize(magnitude, (4, 4)).flatten()

def run():
    print("\n--- [DEV 3] Starting Mid-Level Feature Engineering Extraction ---")
    X_prep = np.load('X_train_features_5.npy')
    
    hog_feats, lbp_feats, edge_feats = [], [], []
    for img in X_prep:
        hog_feats.append(extract_hog(img))
        lbp_feats.append(extract_lbp(img))
        edge_feats.append(extract_edges(img))
        
    # Compile the 6 specific handcrafted combinations requested by the goals schema
    features_dict = {
        "HOG": np.array(hog_feats),
        "LBP": np.array(lbp_feats),
        "Edge_Det": np.array(edge_feats),
        "HOG_LBP": np.hstack((hog_feats, lbp_feats)),
        "LBP_Edge": np.hstack((lbp_feats, edge_feats)),
        "HOG_LBP_Edge": np.hstack((hog_feats, lbp_feats, edge_feats))
    }
    
    np.save('features_dict.npy', features_dict, allow_pickle=True)
    print(f"[DEV 3] Feature extraction matrix build finished. Extracted vector forms for 6 target setups.")

if __name__ == "__main__":
    run()