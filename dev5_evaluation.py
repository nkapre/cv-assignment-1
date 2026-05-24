import os
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

def run():
    print("\n--- [DEV 5] Starting Validation Metrics Report ---")
    trained_payload = np.load('trained_payload.npy', allow_pickle=True).item()
    eval_data = np.load('evaluation_split.npy', allow_pickle=True).item()
    
    X_test = eval_data["X_test"]
    y_test = eval_data["y_test"]
    
    print("\n--- FINAL SCORING METRICS SUMMARY REPORT ---")
    for name, clf in trained_payload.items():
        preds = clf.predict(X_test)
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average='macro')
        print(f"Algorithm: {name:<20} | Test Accuracy: {acc:.4f} | Test F1-Macro: {f1:.4f}")
        
    print("\n[DEV 5] Standby for Student Handwritten Real Target Inference checking module...")
    if not os.path.exists("my_handwritten_letter.png"):
        print("  -> Placeholder notice: Please place your own handwriting file at 'my_handwritten_letter.png' to test external predictions.")

if __name__ == "__main__":
    run()