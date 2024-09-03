from segment_anything import SamPredictor, sam_model_registry
import cv2
import time
import numpy as np
import torch
import matplotlib.pyplot as plt


def remove_black_background(image):
    # Make a True/False mask of pixels whose BGR values sum to more than zero
    alpha = np.sum(image, axis=-1) > 0

    # Convert True/False to 0/255 and change type to "uint8" to match "na"
    alpha = np.uint8(alpha * 255)

    # Stack new alpha layer with existing image to go from BGR to BGRA, i.e. 3 channels to 4 channels
    return np.dstack((na, alpha)) 

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([0, 0, 0, 0.999])
        #color = np.array([0, 0, 0, 0])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=100):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))

def generate_masked_image(image_location, output_prefix):
    sam = sam_model_registry["default"](checkpoint="./sam_vit_h_4b8939.pth")

    image = cv2.imread(image_location)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


    #input_point = np.array([
    #    [850, 1000],
    #    #[500, 1000],
    #    #[200, 1900]
    #])

    input_point = np.array([
        [850, 1000],
        [500, 1000],
        [200, 1900],
        [1590, 3650],
        [2000, 3450]
    ])
    input_label = np.array([1, 1, 1, 1, 1])

    start = time.time()
    predictor = SamPredictor(sam)
    end = time.time()
    print(f"instantiate predictor time: {end - start}")
    predictor.set_image(image)
    end = time.time()
    print(f"set_image time: {end - start}")

    masks, _, _ = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        box=None,
        multimask_output=True
    )

    plt.figure(figsize=(10,10))
    plt.imshow(image)
    #show_points(input_point, input_label, plt.gca())
    plt.savefig(f'/home/Student/s4842338/segment-anything/images/priority_segment_plt_points.png')
    plt.axis('on')

    i = 0
    for mask in masks:
        i += 1
        #color_mask = np.zeros_like(image)
        #color_mask[mask > 0.5] = [255, 255, 255]
        #masked_image = cv2.addWeighted(image, 0.2, color_mask, 0.9, 0)
        show_mask(mask, plt.gca())
        plt.savefig(f'/home/Student/s4842338/segment-anything/images/{output_prefix}_segment_plt_points_masked_image.png')
        #cv2.imwrite(f'/home/Student/s4842338/segment-anything/images/masked_image_{i}.png', cv2.cvtColor(masked_image, cv2.COLOR_RGB2BGR))

    end = time.time()
    print(f"masks time: {end - start}")
    end = time.time()
    print("DONE!")
    print(f"time: {end - start}")


if __name__ == '__main__':
    for i in range(8,101):
        image_location = f"/home/Student/s4842338/segment-anything/images/Priority1b&c_100MEDIA_034_R7North/Segment_{i}_Priority1b&c_100MEDIA.jpg"
        output_prefix = f'segment_{i}_priority_1b_c'
        generate_masked_image(image_location, output_prefix)

        image_location = f"/home/Student/s4842338/segment-anything/images/Priority1b&c_100MEDIA_034_R7North/Segment_{i}_034_R7North.jpg"
        output_prefix = f'segment_{i}_priority_r7_north'
        generate_masked_image(image_location, output_prefix)
