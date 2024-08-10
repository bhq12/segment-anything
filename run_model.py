from segment_anything import SamPredictor, sam_model_registry
import cv2
import time

sam = sam_model_registry["default"](checkpoint="./sam_vit_h_4b8939.pth")

img = cv2.imread("/home/Student/s4842338/dog.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

start = time.time()
predictor = SamPredictor(sam)
end = time.time()
print(f"instantiate predictor time: {end - start}")
predictor.set_image(img)
end = time.time()
print(f"set_image time: {end - start}")
for i in range(1000):
    masks, _, _ = predictor.predict()
end = time.time()
print(f"masks time: {end - start}")
end = time.time()
print("DONE!")
print(f"time: {end - start}")
