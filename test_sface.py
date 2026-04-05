import os, cv2
import model

# Load DB Embeddings
db = model.load_student_sface_embeddings()
if not db:
    print("DB is empty")
    exit()

r = model.get_sface_recognizer()
print(f"Loaded DB: {len(db)} students with {sum(len(v) for v in db.values())} embeddings")

for student in db.keys():
    print(f"\n--- Testing 5 random pics for true student {student} ---")
    folder = os.path.join('dataset', student)
    files = os.listdir(folder)[30:35]
    for fn in files:
        img = cv2.imread(os.path.join(folder, fn))
        if img is None: continue
        live_emb = model.get_sface_embedding(img)
        if live_emb is None: continue
        
        # Test against DB
        best_score = 0
        best_roll = None
        for roll, embs in db.items():
            for ref_emb in embs:
                score = r.match(live_emb, ref_emb, cv2.FaceRecognizerSF_FR_COSINE)
                if score > best_score:
                    best_score = score
                    best_roll = roll
        print(f"Pred: {best_roll:<6} | True: {student:<6} | Cosine: {best_score:.3f}")
