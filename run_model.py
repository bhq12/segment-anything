from segment_anything import sam_model_registry
sam = sam_model_registry["default"](checkpoint="./sam_vit_h_4b8939.pth")
