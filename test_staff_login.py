import sys
import os
import cv2
sys.path.append(r'c:\Users\Admin\Desktop\FaceTrack AI')
from model import verify_staff_with_lbph

staff_folder = r'c:\Users\Admin\Desktop\FaceTrack AI\staff_dataset\srinu'
test_image_path = os.path.join(staff_folder, '1773484885.121601_0.jpg')

# Read image as bytes
with open(test_image_path, 'rb') as f:
    image_bytes = f.read()

# Test 
verified, sim_pct, face_detected = verify_staff_with_lbph(image_bytes, staff_folder, max_train=30, max_distance=105.0)
print(f"Verified: {verified}, Sim PCT: {sim_pct}, Face Detected: {face_detected}")
