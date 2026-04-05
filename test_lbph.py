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

print(f"Training LBPH on {len(faces)} faces...")
recognizer.train(faces, np.array(labels, dtype=np.int32))

# Test
def test_folder(folder, true_label):
    correct = 0
    total = 0
    for f in os.listdir(folder)[20:25]:
        img = cv2.imread(os.path.join(folder, f))
        if img is None: continue
        roi = model._extract_face_roi(img)
        if roi is not None:
            label, dist = recognizer.predict(roi)
            pred_name = label_map.get(label, "Unknown")
            print(f"Pred: {pred_name} (Dist: {dist:.1f}) | True: {label_map[true_label]}")
            if label == true_label: correct += 1
            total += 1
    return correct, total

c1, t1 = test_folder(os.path.join('dataset', datasets[0]), 0)
c2, t2 = test_folder(os.path.join('dataset', datasets[1]), 1)
print(f"Accuracy: {c1+c2}/{t1+t2}")
