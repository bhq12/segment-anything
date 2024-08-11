from segment_anything import SamPredictor, sam_model_registry
import cv2
import time
import numpy as np
import torch
import matplotlib.pyplot as plt

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))


sam = sam_model_registry["default"](checkpoint="./sam_vit_h_4b8939.pth")

image = cv2.imread("/home/Student/s4842338/segment-anything/images/DJI_20230823150505_0771_JPG.rf.8faf22df66126307d64b3a7bfae283e8.jpg")
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
box_1 = np.array([1185, 463, 71.957, 71.265])
box_2 = np.array([1235, 569, 78.876, 85.795])
box_3 = np.array([1308, 648, 56.735, 65.038])
box_4 = np.array([1476, 1091, 75.416, 70.573])
box_5 = np.array([1735, 1262, 92.022, 99.632])

#input_boxes = torch.tensor([box_1, box_2, box_3], device=predictor.device)
input_boxes = torch.tensor(
    [box_1, box_2, box_3, box_4, box_5],
    device=predictor.device
)

transformed_boxes = predictor.transform.apply_boxes_torch(input_boxes, image.shape[:2])  
masks, _, _ = predictor.predict_torch(
    point_coords=None,
    point_labels=None,
    boxes=transformed_boxes,
    multimask_output=True
)



#masks, _, _ = predictor.predict(
#    point_coords=None,
#    point_labels=None,
#    box=box_1,
#    multimask_output=True
#)
#
plt.figure(figsize=(10,10))
plt.imshow(image)
show_box(box_1, plt.gca())
plt.savefig(f'/home/Student/s4842338/segment-anything/images/plt_box_1.png')
show_box(box_2, plt.gca())
plt.savefig(f'/home/Student/s4842338/segment-anything/images/plt_box_2.png')
show_box(box_3, plt.gca())
plt.savefig(f'/home/Student/s4842338/segment-anything/images/plt_box_3.png')
show_box(box_4, plt.gca())
plt.savefig(f'/home/Student/s4842338/segment-anything/images/plt_box_4.png')
show_box(box_5, plt.gca())
plt.savefig(f'/home/Student/s4842338/segment-anything/images/plt_box_5.png')
plt.axis('on')

i = 0
for mask in masks:
    i += 1
    #color_mask = np.zeros_like(image)
    #color_mask[mask > 0.5] = [255, 255, 255]
    #masked_image = cv2.addWeighted(image, 0.2, color_mask, 0.9, 0)
    show_mask(mask, plt.gca())
    plt.savefig(f'/home/Student/s4842338/segment-anything/images/plt_masked_image_{i}.png')
    #cv2.imwrite(f'/home/Student/s4842338/segment-anything/images/masked_image_{i}.png', cv2.cvtColor(masked_image, cv2.COLOR_RGB2BGR))

end = time.time()
print(f"masks time: {end - start}")
end = time.time()
print("DONE!")
print(f"time: {end - start}")
