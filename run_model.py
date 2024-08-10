from segment_anything import SamPredictor, sam_model_registry
import cv2
import time
import numpy as np
import torch

sam = sam_model_registry["default"](checkpoint="./sam_vit_h_4b8939.pth")

image = cv2.imread("/home/Student/s4842338/segment-anything/images/DJI_20230823145345_0089_JPG.rf.65724ef904d2a04e9dacbbf154b44297.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

start = time.time()
predictor = SamPredictor(sam)
end = time.time()
print(f"instantiate predictor time: {end - start}")
predictor.set_image(image)
end = time.time()
print(f"set_image time: {end - start}")
#
#box_1 = np.array([573, 966, 73.741, 85.576])
#box_2 = np.array([601, 778, 60.996, 101.053])
#box_3 = np.array([512, 789, 39.147, 40.967])
#input_boxes = torch.tensor([box_1, box_2, box_3], device=predictor.device)
#transformed_boxes = predictor.transform.apply_boxes_torch(input_boxes, image.shape[:2])  
#masks, _, _ = predictor.predict_torch(
#    point_coords=None,
#    point_labels=None,
#    boxes=transformed_boxes,
#    multimask_output=True
#)

box_1 = np.array([573, 966, 73.741, 85.576])

masks, _, _ = predictor.predict(
    point_coords=None,
    point_labels=None,
    box=box_1,
    multimask_output=True
)
i = 0
for mask in masks:
    i += 1
    color_mask = np.zeros_like(image)
    color_mask[mask > 0.5] = [30, 144, 255]
    masked_image = cv2.addWeighted(image, 0.6, color_mask, 0.4, 0)
    cv2.imwrite(f'/home/Student/s4842338/segment-anything/images/masked_image_{i}.png', cv2.cvtColor(masked_image, cv2.COLOR_RGB2BGR))

end = time.time()
print(f"masks time: {end - start}")
end = time.time()
print("DONE!")
print(f"time: {end - start}")
