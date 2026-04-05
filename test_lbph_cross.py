import os, cv2, numpy as np
import model

datasets = [d for d in os.listdir('dataset') if os.path.isdir(os.path.join('dataset', d))]
if len(datasets) < 2:
    print("Need at least 2 students.")
    exit()

# Train on ONLY Person 0
recognizer = cv2.face.LBPHFaceRecognizer_create()
faces = []
labels = []

d = datasets[0]
folder = os.path.join('dataset', d)
for f in os.listdir(folder)[:20]:
    img = cv2.imread(os.path.join(folder, f))
    if img is None: continue
    roi = model._extract_face_roi(img)
    if roi is not None:
        faces.append(roi)
        labels.append(0)

recognizer.train(faces, np.array(labels, dtype=np.int32))

print(f"Model trained ONLY on {d}")

def print_test(folder, person_name):
    print(f"\n--- Testing {person_name} images against {d} model ---")
    files = os.listdir(folder)[20:30]
    for fn in files:
        img = cv2.imread(os.path.join(folder, fn))
        if img is None: continue
        roi = model._extract_face_roi(img)
        if roi is not None:
            label, dist = recognizer.predict(roi)
            print(f"Dist to {d}: {dist:.1f}")

print_test(os.path.join('dataset', datasets[0]), datasets[0])
print_test(os.path.join('dataset', datasets[1]), datasets[1])
