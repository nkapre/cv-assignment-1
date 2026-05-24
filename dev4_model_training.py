import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

def run():
    print("\n--- [DEV 4] Starting Model Training & Validation ---")
    features_dict = np.load('features_dict.npy', allow_pickle=True).item()
    y = np.load('y_zero_indexed.npy')
    
    # Isolate optimized combination from the generated dictionary
    X_combined = features_dict["HOG_LBP_Edge"]
    
    # 80-20 Train/Test Segment Split
    X_train, X_test, y_train, y_test = train_test_split(X_combined, y, test_size=0.20, stratify=y, random_state=42)
    
    # Standardize via Z-Score Scaling transform
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train the core targeted classical classifier model configurations
    print("[DEV 4] Training multiple classical algorithms sequentially...")
    models = {
        "Random_Forest": RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1),
        "SVM": SVC(kernel='linear', C=1.0, random_state=42),
        "k-NN": KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
        "Logistic_Regression": LogisticRegression(max_iter=100, solver='saga', random_state=42, n_jobs=-1)
    }
    
    trained_payload = {}
    for name, clf in models.items():
        clf.fit(X_train_scaled, y_train)
        trained_payload[name] = clf
        print(f"  [✔] Finished training {name} model.")
        
    np.save('trained_payload.npy', trained_payload, allow_pickle=True)
    np.save('evaluation_split.npy', {"X_test": X_test_scaled, "y_test": y_test, "scaler": scaler}, allow_pickle=True)
    print("[DEV 4] Finished fitting model payload arrays.")

if __name__ == "__main__":
    run()