import os, cv2, numpy as np
import model

datasets = [d for d in os.listdir('dataset') if os.path.isdir(os.path.join('dataset', d))]
if len(datasets) < 2:
    print("Need at least 2 students.")
    exit()

def get_embs(folder):
    embs = []
    for f in os.listdir(folder)[:5]:
        img = cv2.imread(os.path.join(folder, f))
        emb = model._extract_landmark_embedding(img, model._get_face_mesh())
        if emb is not None: embs.append(emb)
    return embs

embs_subj1 = get_embs(os.path.join('dataset', datasets[0]))
embs_subj2 = get_embs(os.path.join('dataset', datasets[1]))

if len(embs_subj1) > 0 and len(embs_subj2) > 0:
    for i in range(len(embs_subj1)):
        dist_self = np.linalg.norm(embs_subj1[0] - embs_subj1[i])
        print(f"Euclidean dist {datasets[0]} self photo {i}: {dist_self:.4f}")
    
    for i in range(len(embs_subj2)):
        dist_other = np.linalg.norm(embs_subj1[0] - embs_subj2[i])
        print(f"Euclidean dist {datasets[0]} vs {datasets[1]} photo {i}: {dist_other:.4f}")
