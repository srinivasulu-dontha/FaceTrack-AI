import os, cv2, numpy as np
import model

datasets = [d for d in os.listdir('dataset') if os.path.isdir(os.path.join('dataset', d))]
if len(datasets) < 2:
    print("Need at least 2 students.")
    exit()

recognizer = cv2.face.LBPHFaceRecognizer_create()
faces = []
labels = []
label_map = {}
label_id = 0

for d in datasets:
    folder = os.path.join('dataset', d)
    label_map[label_id] = d
    for f in os.listdir(folder)[:20]:
        img = cv2.imread(os.path.join(folder, f))
        if img is None: continue
        roi = model._extract_face_roi(img)
        if roi is not None:
            faces.append(roi)
            labels.append(label_id)
    label_id += 1

recognizer.train(faces, np.array(labels, dtype=np.int32))

def print_test(folder, true_label):
    print(f"\n--- Testing {label_map[true_label]} images ---")
    files = os.listdir(folder)[20:30]
    for fn in files:
        img = cv2.imread(os.path.join(folder, fn))
        if img is None: continue
        roi = model._extract_face_roi(img)
        if roi is not None:
            label, dist = recognizer.predict(roi)
            pred_name = label_map.get(label, "Unknown")
            is_match = "MATCH" if label == true_label else "MISMATCH"
            print(f"Pred: {pred_name:<10} | True: {label_map[true_label]:<10} | Dist: {dist:.1f} | {is_match}")

print_test(os.path.join('dataset', datasets[0]), 0)
print_test(os.path.join('dataset', datasets[1]), 1)
