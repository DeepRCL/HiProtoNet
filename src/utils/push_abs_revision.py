import torch
import numpy as np
import cv2
import os
import time
import random

from src.utils.explainability_utils import get_src, get_normalized_upsample_occurence_maps, get_heatmap
from src.utils.utils import makedir, save_pickle
from src.utils.video_utils import saveVideo
from tqdm import tqdm
# from moviepy.video.io.bindings import mplfig_to_npimage
# import matplotlib.pyplot as plt
import imageio
import warnings


def create_colorbar(height, width, min_val, max_val, grid_orientation='horizontal'):
    """
        Create a colorbar once to be reused for all frames.
        Returns the array in [BGR] format.
    """
    colorbar = np.zeros((height, width, 3), dtype=np.uint8)
    if grid_orientation == 'horizontal':
        # color bar has to be vertical
        max_val_org = (2, 10)
        min_val_org = (2, height - 5)
        for i in range(height):
            # Apply color map with explicit orientation in BGR format
            # idx 0 = top pixel = max value = red
            color_value = np.array([[int(255 * (height - i - 1) / height)]], dtype=np.uint8)
            colorbar[i, :, :] = cv2.applyColorMap(color_value, cv2.COLORMAP_TURBO)[0, 0]

    elif grid_orientation == 'vertical':
        # color bar has to be horizontal
        max_val_org = (10, height // 2)
        min_val_org = (width // 5 * 4, height // 2)
        for i in range(width):
            # Apply color map with explicit orientation in BGR format
            # idx 0 = left most pixel = max value = red
            color_value = np.array([[int(255 * (width - i - 1) / width)]], dtype=np.uint8)
            colorbar[:, i, :] = cv2.applyColorMap(color_value, cv2.COLORMAP_TURBO)[0, 0]

    # Add min/max values
    cv2.putText(colorbar, f"{max_val:.2f}", max_val_org, cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
    cv2.putText(colorbar, f"{min_val:.2f}", min_val_org, cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
    return colorbar


def prototype_plot_frame_optimized(images_dict, occ_map, occ_map_min, occ_map_max, proto_id, fn, pred, gt,
                                   colorbar=None,
                                   grid_orientation='horizontal'):
    """
        Optimized version that works with numpy arrays directly instead of matplotlib.
        Accepts the arrays in [RGB] format, turns them to [BGR] to great the grid, returns the array in [BGR] format.
    """
    # convert RGB to BGR for visualization with CV2
    base_img = cv2.cvtColor(images_dict["base"], cv2.COLOR_RGB2BGR)
    masked_img = cv2.cvtColor(images_dict["masked"], cv2.COLOR_RGB2BGR)
    overlay_img = cv2.cvtColor(images_dict["overlay"], cv2.COLOR_RGB2BGR)

    # Normalize occurrence map for visualization
    norm_occ_map = ((occ_map - occ_map_min) / (occ_map_max - occ_map_min) * 255).astype(np.uint8)
    # Apply color map on occurrence map, format is BGR
    occ_map_colored = cv2.applyColorMap(norm_occ_map, cv2.COLORMAP_TURBO)  # [BGR]

    if grid_orientation == 'horizontal':
        # Stack images horizontally with colorbar! all should be in BGR
        occ_map_with_colorbar = np.hstack([occ_map_colored, colorbar])
        frame = np.hstack([base_img, masked_img, overlay_img, occ_map_with_colorbar])
        buffer_shape = (40, frame.shape[1], 3)

    elif grid_orientation == 'vertical':
        # Stack images vertically with colorbar! all should be in BGR
        occ_map_with_colorbar = np.vstack([occ_map_colored, colorbar])  # [BGR]
        frame = np.vstack([base_img, masked_img, overlay_img, occ_map_with_colorbar])  # [BGR]
        buffer_shape = (40, frame.shape[0], 3)

    # Add black buffer at the top(left) for the horizontal(vertical) grid orientation
    buffer = np.zeros(buffer_shape, dtype=np.uint8)  # horizontal first

    # Add text in the buffer zone
    text1 = f"p_{proto_id:02d} | gt={gt} | pred = {[f'{pred[i]:.2f}' for i in range(pred.shape[0])]}"
    text2 = f"{fn}"
    # Add text in the buffer zone  (horizontal orientation = normal)
    cv2.putText(buffer, text1, (10, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    cv2.putText(buffer, text2, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    if grid_orientation == 'horizontal':
        # Add black buffer with text at the top
        frame = np.vstack([buffer, frame])
    elif grid_orientation == 'vertical':
        # Rotate the text image by 90 degrees
        buffer = cv2.rotate(buffer, cv2.ROTATE_90_CLOCKWISE)
        # Add black buffer with text at the left side
        frame = np.hstack([buffer, frame])

    return frame


def prototype_plot(unnorm_img, upsampled_occurrence_map, prototypes_src_imgs_masked,
                  prototypes_overlayed_imgs, proto_id, fn, pred, gt, proto_dir, interp="none",
                  grid_orientation='horizontal'):
    D = len(unnorm_img.shape)

    # Create dictionary of images
    images_dict = {
        "base": unnorm_img,  # [RGB]
        "masked": prototypes_src_imgs_masked,  # [RGB]
        "overlay": prototypes_overlayed_imgs  # [RGB]
    }

    # Convert to uint8 for both image and video
    for key in images_dict:
        if images_dict[key].dtype != np.uint8:
            images_dict[key] = (images_dict[key] * 255).astype(np.uint8)

    occ_min = np.amin(upsampled_occurrence_map)
    occ_max = np.amax(upsampled_occurrence_map) + 1e-7

    # Create colorbar once
    if D == 3:
        height = unnorm_img.shape[0]  # Use image height
        width = unnorm_img.shape[1]  # Use image width
    else:  # D == 4
        height = unnorm_img.shape[1]  # Use video frame height
        width = unnorm_img.shape[2]  # Use video frame width

    if grid_orientation == 'horizontal':
        width = 30
    elif grid_orientation == 'vertical':
        height = 30
    colorbar = create_colorbar(
        height=height,
        width=width,
        min_val=occ_min,
        max_val=occ_max,
        grid_orientation=grid_orientation,
    )   # [BGR]

    if D == 3:
        # Single image case
        frame = prototype_plot_frame_optimized(
            images_dict,
            upsampled_occurrence_map,
            occ_min, occ_max,
            proto_id, fn, pred, gt,
            colorbar=colorbar,
            grid_orientation=grid_orientation,
        )
        # Save as PNG
        cv2.imwrite(os.path.join(proto_dir, f"{proto_id:02d}_{fn}.png"), frame)   # [BGR]

    elif D == 4:
        # Video case - process all frames
        frames = [
            prototype_plot_frame_optimized(
                {k: v[t] for k, v in images_dict.items()},
                upsampled_occurrence_map[t],
                occ_min, occ_max,
                proto_id, fn, pred, gt,
                colorbar=colorbar,
                grid_orientation=grid_orientation,
            )
            for t in tqdm(range(unnorm_img.shape[0]))
        ]

        # Pad frames and save video
        frames = [pad_to_16(frame) for frame in frames]   # [BGR]
        frames_rgb = [frame[..., ::-1] for frame in frames]  # Convert BGR to RGB using slicing
        video_path = os.path.join(proto_dir, f"{proto_id:02d}-file_{fn}.mp4")
        imageio.mimwrite(video_path, frames_rgb, fps=30, codec='libx264', quality=7)   # [RGB]


def pad_to_16(frame):
    """Pad frame dimensions to be multiples of 16 for H.264 encoding"""
    h, w = frame.shape[:2]
    new_h = ((h + 15) // 16) * 16
    new_w = ((w + 15) // 16) * 16

    padded = np.zeros((new_h, new_w, 3), dtype=np.uint8)
    padded[:h, :w] = frame
    return padded


def push_prototypes(
    dataloader,  # pytorch dataloader
    # dataset,   # pytorch dataset for train_push group
    # prototype_layer_stride=1,
    model,  # pytorch network with feature encoder and prototype vectors
    class_specific=True,  # enable pushing protos from only the alotted class
    abstain_class=True,  # indicates K+1-th class is of the "abstain" type
    preprocess_input_function=None,  # normalize if needed
    root_dir_for_saving_prototypes=None,  # if not None, prototypes will be saved in this dir
    epoch_number=None,  # if not provided, prototypes saved previously will be overwritten
    log=print,
    prototype_img_filename_prefix=None,
    prototype_self_act_filename_prefix=None,
    proto_bound_boxes_filename_prefix=None,
    replace_prototypes=True,
    push_local_prots=False,  # TODO use this to push local prototypes to closest local features!
):
    """
    Search the training set for image patches that are semantically closest to
    each learned prototype, then updates the prototypes to those image patches.

    To do this, it computes the image patch embeddings (IPBs) and saves those
    closest to the prototypes. It also saves the prototype-to-IPB distances and
    predicted occurrence maps.

    If abstain_class==True, it assumes num_classes actually equals to K+1, where
    K is the number of real classes and 1 is the extra "abstain" class for
    uncertainty estimation.
    """
    if push_local_prots:
        assert(model.local_prototype_method == "attn")

    model.eval()
    if push_local_prots:
        log(f"############## push local prototypes at epoch {epoch_number} #################")
    else:
        log(f"############## push at epoch {epoch_number} #################")

    if replace_prototypes:
        log("\t ++++ Replacing prototypes ...")
    else:
        log("\t ---- Not replacing prototypes ...")

    start = time.time()

    # creating the folder (with epoch number) to save the prototypes' info and visualizations
    if root_dir_for_saving_prototypes != None:
        if epoch_number != None:
            proto_epoch_dir = os.path.join(root_dir_for_saving_prototypes, "epoch-" + str(epoch_number))
            makedir(proto_epoch_dir)
        else:
            proto_epoch_dir = root_dir_for_saving_prototypes
    else:
        proto_epoch_dir = None

    # find the number of prototypes, and number of classes for this push
    if push_local_prots:
        prototype_shape = model.local_prototype_shape  # shape (P, D, (1), 1, 1)
        P = model.num_local_prototypes
        proto_class_identity = np.argmax(model.local_prototype_class_identity.cpu().numpy(), axis=1)  # shape (PL)
    else:
        prototype_shape = model.prototype_shape  # shape (P, D, (1), 1, 1)
        P = model.num_prototypes
        proto_class_identity = np.argmax(model.prototype_class_identity.cpu().numpy(), axis=1)  # shape (P)
    proto_class_specific = np.full(P, class_specific)
    num_classes = model.num_classes
    P_per_class = P // num_classes
    if abstain_class:
        K = num_classes - 1
        assert K >= 2, "Abstention-push must have >= 2 classes not including abstain"
        # for the uncertainty prototypes, class_specific is False
        # for now assume that each class (inc. unc.) has P_per_class == P/num_classes
        proto_class_specific[K * P_per_class : P] = False
    else:
        K = num_classes

    # keep track of the input embedding closest to each prototype
    proto_dist_ = np.full(P, np.inf)  # saves the distances to prototypes (distance = 1-CosineSimilarities). shape (P)
    # save some information dynamically for each prototype
    # which are updated whenever a closer match to prototype is found
    occurrence_map_ = [None for _ in range(P)]  # saves the computed occurence maps. shape (P, 1, (T), H, W)
    # saves the input to prototypical layer (conv feature * occurrence map), shape (P, D)
    protoL_input_ = [None for _ in range(P)]
    # saves the input images with embeddings closest to each prototype. shape (P, 3, (To), Ho, Wo)
    image_ = [None for _ in range(P)]
    # saves the gt label. shape (P)
    gt_ = [None for _ in range(P)]
    # saves the prediction logits of cases seen. shape (P, K)
    pred_ = [None for _ in range(P)]
    # saves the filenames of cases closest to each prototype. shape (P)
    filename_ = [None for _ in range(P)]

    data_iter = iter(dataloader)
    iterator = tqdm(range(len(dataloader)), dynamic_ncols=True)
    for push_iter in iterator:
        data_sample = next(data_iter)

        x = data_sample["cine"]  # shape (B, 3, (To), Ho, Wo)
        if preprocess_input_function is not None:
            x = preprocess_input_function(x)

        # get the network outputs for this instance
        with (torch.no_grad()):
            x = x.cuda()
            returned = model.push_forward(x)
            if len(returned) == 4:  # for euclidean models
                (
                    protoL_input_torch,
                    proto_dist_torch,
                    occurrence_map_torch,
                    logits,
                ) = returned
            elif len(returned) == 7:  # for hyperbolic models
                if push_local_prots:  # L is 1 when  local_feat_method='1_per_input' and is PL when 1_per_class
                    (_, _, _, protoL_input_torch, proto_dist_torch, occurrence_map_torch, logits) = returned
                    if model.cls_method == "both":
                        # TODO decide on what to show on the prototypes? the logits of its own level? or overal?
                        logits = logits[1]
                        # logits = 0.5 * (logits[0] + logits[1])
                else: # shape (N, L, D)   # shape (N, PL)  # shape (N, L, 1, H, W)
                    (protoL_input_torch, proto_dist_torch, occurrence_map_torch, _, _, _, logits) = returned
                    if model.cls_method == "both":
                        # TODO decide on what to show on the prototypes? the logits of its own level? or overal?
                        logits = logits[0]
                        # logits = 0.5 * (logits[0] + logits[1])

            # pred_torch = logits.softmax(dim=1)

        # record down batch data as numpy arrays
        protoL_input = protoL_input_torch.detach().cpu().numpy()  # shape (B, P, D)
        proto_dist = proto_dist_torch.detach().cpu().numpy()  # shape (B, P)
        occurrence_map = occurrence_map_torch.detach().cpu().numpy()  # shape (B, P, 1, (T), H, W)
        # pred = pred_torch.detach().cpu().numpy() # shape (B, num_classes)
        pred = logits.detach().cpu().numpy()  # shape (B, num_classes)
        gt = data_sample["target_AS"].detach().cpu().numpy()  # shape (B)
        image = x.detach().cpu().numpy()  # shape (B, 3, (To), Ho, Wo)
        filename = data_sample["filename"]  # shape (B)

        # for each prototype, find the minimum distance and their indices
        for j in range(P):
            proto_dist_j = proto_dist[:, j]  # (B)
            if proto_class_specific[j]:
                # compare with only the images of the prototype's class
                proto_dist_j = np.ma.masked_array(proto_dist_j, gt != proto_class_identity[j])
                if proto_dist_j.mask.all():
                    # if none of the classes this batch are the class of interest, move on
                    continue
            proto_dist_j_min = np.amin(proto_dist_j)  # scalar

            # if the distance this batch is smaller than prev.best, save it
            if proto_dist_j_min <= proto_dist_[j]:
                a = np.argmin(proto_dist_j)
                proto_dist_[j] = proto_dist_j_min
                if push_local_prots:
                    if protoL_input.shape[1] == 1 and model.local_feat_method == "1_per_input":
                        protoL_input_[j] = protoL_input[a, 0]
                        occurrence_map_[j] = occurrence_map[a, 0]
                    elif model.local_feat_method == "1_per_class":
                        protoL_input_[j] = protoL_input[a, j]
                        occurrence_map_[j] = occurrence_map[a, j//model.num_local_prototypes_per_class]
                else:
                    protoL_input_[j] = protoL_input[a, j]
                    occurrence_map_[j] = occurrence_map[a, j]
                pred_[j] = pred[a]
                image_[j] = image[a]
                gt_[j] = gt[a]
                filename_[j] = filename[a]

    prototypes_similarity_to_src_ROIs = 1 - np.array(proto_dist_)  # invert distance to similarity  shape (P)
    prototypes_occurrence_maps = np.array(occurrence_map_)  # shape (P, 1, (T), H, W)
    prototypes_src_imgs = np.array(image_)  # shape (P, 3, (To), Ho, Wo)
    prototypes_gts = np.array(gt_)  # shape (P)
    prototypes_preds = np.array(pred_)  # shape (P, K)
    prototypes_filenames = np.array(filename_)  # shape (P)

    if push_local_prots:
        prototypes_features_to_be_replaced = model.local_prototype_vectors.data.squeeze().cpu().numpy()
    else:
        prototypes_features_to_be_replaced = model.prototype_vectors.data.squeeze().cpu().numpy()
    # save the prototype information in a pickle file
    prototype_data_dict = {
        "prototypes_filenames": prototypes_filenames,
        "prototypes_src_imgs": prototypes_src_imgs,
        "prototypes_post_push":  np.array(protoL_input_),
        "prototypes_pre_push":  prototypes_features_to_be_replaced,
        "prototypes_gts": prototypes_gts,
        "prototypes_preds": prototypes_preds,
        "prototypes_occurrence_maps": prototypes_occurrence_maps,
        "prototypes_similarity_to_src_ROIs": prototypes_similarity_to_src_ROIs,
        "prototypes_distance_to_src_ROIs": proto_dist_,
    }
    # check if lift_prototypes is available in the model, then add them to the dictionary
    if hasattr(model, "lift_prototypes"):
        prototype_data_dict.update({
            "lift_prototypes": model.lift_prototypes,
            "curv": model.curv.data.item(),
            "visual_alpha": model.visual_alpha.data.item(),
            "_curv_minmax": model._curv_minmax,
        })
    save_pickle(prototype_data_dict, f"{proto_epoch_dir}/prototypes_info.pickle")

    #########################################
    # perform visualization for each prototype
    log("\tVisualizing prototypes ...")
    #########################################
    ##### Process the prototype info ########
    # get the source image/video
    prototypes_src_imgs, upsampler = get_src(prototypes_src_imgs)  # shape (P, (To), Ho, Wo, 3)

    # resize, upsample, and normalize the occurrence map.  shape (P, (To), Ho, Wo)
    rescaled_occurrence_maps, upsampled_occurrence_maps = get_normalized_upsample_occurence_maps(
        prototypes_occurrence_maps, upsampler)  # shape (P, (To), Ho, Wo)

    # # prototype src image masked with normalized occurrence map
    mask = np.expand_dims(rescaled_occurrence_maps, axis=-1) # shape (P, (T), H, W, 1)
    prototypes_src_imgs_masked = prototypes_src_imgs * mask  # shape (P, (To), Ho, Wo, 3)

    # prototype src image with normalized occurrence map overlay
    prots_heatmaps = get_heatmap(rescaled_occurrence_maps)  # shape (P, (To), Ho, Wo, 3)   # [RGB]
    # prototypes_overlayed_imgs = np.clip(prototypes_src_imgs + 0.3 * prots_heatmaps, 0, 1)  # shape (P, (To), Ho, Wo, 3)
    prototypes_overlayed_imgs = np.clip(0.7 *prototypes_src_imgs + 0.3 * prots_heatmaps, 0, 1)  # shape (P, (To), Ho, Wo, 3)

    for j in range(P):
        if image_[j] is not None:
            prototype_plot(unnorm_img=prototypes_src_imgs[j], upsampled_occurrence_map=upsampled_occurrence_maps[j],
                           prototypes_src_imgs_masked=prototypes_src_imgs_masked[j],
                           prototypes_overlayed_imgs=prototypes_overlayed_imgs[j], proto_id=j, fn=filename_[j],
                           pred=pred_[j], gt=gt_[j], proto_dir=proto_epoch_dir,
                           interp="bilinear",   # 'none', 'nearest', 'bilinear', 'auto'
                           grid_orientation='horizontal')
    if replace_prototypes:
        protoL_input_ = np.array(protoL_input_)
        log("\tExecuting push ...")
        prototype_update = np.reshape(protoL_input_, tuple(prototype_shape))
        if push_local_prots:
            model.local_prototype_vectors.data.copy_(torch.tensor(prototype_update, dtype=torch.float32).cuda())
        else:
            model.prototype_vectors.data.copy_(torch.tensor(prototype_update, dtype=torch.float32).cuda())
    end = time.time()
    log("\tpush time: \t{0}".format(end - start))
