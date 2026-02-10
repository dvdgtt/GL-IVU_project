import random
import numpy as np
import xarray as xr
from IPython.terminal.shortcuts.auto_suggest import accept
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

def get_phase_and_magnitude(xar_dinsar):
    """
    :param xar_dinsar: dinsar scene. First entry must be real, second must be imaginary
    :return: phase and magnitude
    """
    real = xar_dinsar.values[0, :, :]
    imag = xar_dinsar.values[1, :, :]

    # Create complex
    complex_data = real + 1j * imag
    phase = np.angle(complex_data)
    magnitude = np.abs(complex_data)

    return phase, magnitude


def flow_train_dataset(scene_dinsar, scene_mask, scene_ground_lines, scene_name, patch_size, random_state, path_out, save):

    seen = xr.zeros_like(scene_mask)
    limit = 15000   # upper limit
    no_images_tot = 0   # total img counter
    pbar = tqdm(total=limit)

    while True:

        list_patches, list_masks, seen, indexes_x, indexes_y = create_dataset_train(scene_dinsar,
                                                                                    scene_mask,
                                                                                    scene_ground_lines,
                                                                                    patch_size,
                                                                                    seen,
                                                                                    random_state)

        assert len(list_patches) == len(list_masks) == len(indexes_x) == len(indexes_y), "Error occurred."

        no_images_it = len(list_patches)    # iteration count
        no_images_tot += no_images_it       # total count

        if no_images_it == 0:
            print("Unable to find new patches")
            break

        # SAVE THE IMAGES
        for (patch, mask, idx_x, idx_y) in zip(list_patches, list_masks, indexes_x, indexes_y):

            patch_np = patch.values  # float32 (2, H, W)
            mask_np = mask.values  # uint8 (1, H, W)

            if save:
                # directory per la dimensione (es. .../patches-train/128)
                size_dir = os.path.join(path_out, str(patch_np.shape[1]))
                # directory train{size} (es. .../patches-train/128/train128)
                train_dir = os.path.join(size_dir, f"train{patch_np.shape[1]}")
                os.makedirs(train_dir, exist_ok=True)

                # nome file
                file_out = os.path.join(train_dir, f"{scene_name}_x{idx_x}_y{idx_y}_s{patch_np.shape[1]}")

                # salva .npz (re + im + mask)
                np.savez(f"{file_out}.npz", image=patch_np, mask=mask_np)

                # salva mask .png
                #plt.imsave(f"{file_out}_mask.png", mask_np.squeeze(), cmap="binary")

                # salva phase .png
                phase, magnitude = get_phase_and_magnitude(patch)
                phase_norm = (phase + np.pi) / (2 * np.pi)  # Normalize phase [-π,π] -> [0,1]
                #plt.imsave(f"{file_out}_phase.png", phase_norm, cmap="twilight")


        pbar.update(no_images_it)

        if no_images_tot >= limit:
            print(f"We have exceeded the limit {limit}")
            break

    pbar.close()

    return seen, no_images_tot

def flow_noisy_train_dataset(scene_dinsar, scene_mask, scene_ground_lines, scene_name, patch_size, random_state, path_out, save):

    seen = xr.zeros_like(scene_mask)
    limit = 100   # upper limit
    no_images_tot = 0   # total img counter
    pbar = tqdm(total=limit)

    while True:

        list_patches, list_masks, seen, indexes_x, indexes_y = create_noise_dataset_train(scene_dinsar,
                                                                                    scene_mask,
                                                                                    scene_ground_lines,
                                                                                    patch_size,
                                                                                    seen,
                                                                                    random_state)

        assert len(list_patches) == len(list_masks) == len(indexes_x) == len(indexes_y), "Error occurred."

        no_images_it = len(list_patches)    # iteration count
        no_images_tot += no_images_it       # total count

        if no_images_it == 0:
            print("Unable to find new patches")
            break

        # SAVE THE IMAGES
        for (patch, mask, idx_x, idx_y) in zip(list_patches, list_masks, indexes_x, indexes_y):

            patch_np = patch.values  # float32 (2, H, W)
            mask_np = mask.values  # uint8 (1, H, W)

            if save:
                # directory per la dimensione
                size_dir = os.path.join(path_out, str(patch_np.shape[1]))
                # directory train{size}
                train_dir = os.path.join(size_dir, f"noise_{patch_np.shape[1]}")
                os.makedirs(train_dir, exist_ok=True)

                # nome file
                file_out = os.path.join(train_dir, f"NOISY{scene_name}_x{idx_x}_y{idx_y}_s{patch_np.shape[1]}")

                # salva .npz (re + im + mask)
                np.savez(f"{file_out}.npz", image=patch_np, mask=mask_np)

                # salva mask .png
                plt.imsave(f"{file_out}_mask.png", mask_np.squeeze(), cmap="binary")

                # salva phase .png
                phase, magnitude = get_phase_and_magnitude(patch)
                phase_norm = (phase + np.pi) / (2 * np.pi)  # Normalize phase [-π,π] -> [0,1]
                plt.imsave(f"{file_out}_phase.png", phase_norm, cmap="hsv")


        pbar.update(no_images_it)

        if no_images_tot >= limit:
            print(f"We have exceeded the limit {limit}")
            break

    pbar.close()

    return seen, no_images_tot

def check_seen_percentage(seen_patch, seen_threshold, perc_threshold):
    # ottieni array numpy e rimuovi eventuale asse canale
    arr = seen_patch.values if hasattr(seen_patch, "values") else seen_patch
    if arr.ndim == 3:  # (1,H,W)
        arr = arr[0]
    total = arr.size
    if total == 0:
        return False

    valid = np.sum(arr < seen_threshold)
    perc_valid = valid / total

    if perc_threshold > 1:
        perc_thresh_norm = perc_threshold / 100.0
    else:
        perc_thresh_norm = perc_threshold

    return perc_valid >= perc_thresh_norm


def create_dataset_train(scene_dinsar, scene_mask, scene_ground_lines,
                         patch_size, seen, random_state, max_iter=50000,
                         seen_threshold=3, perc_threshold=40):


    i_h, i_w = scene_dinsar.shape[1:]
    s_h, s_w = scene_mask.shape[1:]
    p_h, p_w = patch_size, patch_size

    if p_h > i_h:
        raise ValueError("Height of the patch should be less than the height of the image.")
    if p_w > i_w:
        raise ValueError("Width of the patch should be less than the width of the image.")
    if i_h != s_h or i_w != s_w:
        raise ValueError("Image and mask must have same shape.")

    size = p_h * p_w
    patches, masks, idx_x, idx_y = [], [], [], []
    seed = None if random_state == -1 else random_state
    random.seed(seed)

    for _ in range(max_iter):
        append = True
        i_s = random.randint(0, i_h - p_h)
        j_s = random.randint(0, i_w - p_w)
        center_i = int(i_s + (p_h / 2))
        center_j = int(j_s + (p_w / 2))

        patch = scene_dinsar[:, i_s:i_s + p_h, j_s:j_s + p_w]
        mask_patch = scene_mask[:, i_s:i_s + p_h, j_s:j_s + p_w]

        # CONDITION 1: vogliamo patch che CONTENGANO almeno un pixel di GL
        if float(mask_patch.sum()) == 0:
            append = False

        # se già scartata per mask, salta i controlli su seen
        if not append:
            continue

        # CONDITION 2: controllo 'seen' — accetta se (poco visto) OR (sufficiente % "freschi")
        # calcola box 32x32 intorno al centro con protezione bordi
        i0 = max(0, center_i - 15)
        i1 = min(seen.shape[1], center_i + 16)   # exclusive upper bound in slicing
        j0 = max(0, center_j - 15)
        j1 = min(seen.shape[2], center_j + 16)

        box_sum = float(seen[:, i0:i1, j0:j1].sum())

        cond1 = box_sum <= 200
        cond2 = check_seen_percentage(seen[:, i_s:i_s + p_h, j_s:j_s + p_w],
                                      seen_threshold=seen_threshold,
                                      perc_threshold=perc_threshold)

        # Se nessuna delle due condizioni è vera, scarta
        if not (cond1 or cond2):
            append = False

        if append:
            assert mask_patch.size == size, "Something wrong to be checked."
            # aggiorna seen sull'intero patch
            seen[:, i_s:i_s + p_h, j_s:j_s + p_w] += 1
            patches.append(patch)
            masks.append(mask_patch)
            idx_x.append(i_s)
            idx_y.append(j_s)

        if len(patches) == 10:
            break

    return patches, masks, seen, idx_x, idx_y

def create_noise_dataset_train(scene_dinsar, scene_mask, scene_grounding_lines, patch_size, seen, random_state, max_iter=50000,
                         seen_threshold=3, perc_threshold=60):

    i_h, i_w = scene_dinsar.shape[1:]
    s_h, s_w = scene_mask.shape[1:]
    p_h, p_w = patch_size, patch_size

    if p_h > i_h:
        raise ValueError("Height of the patch should be less than the height of the image.")
    if p_w > i_w:
        raise ValueError("Width of the patch should be less than the width of the image.")
    if i_h != s_h or i_w != s_w:
        raise ValueError("Image and mask must have same shape.")

    size = p_h * p_w
    patches, masks, idx_x, idx_y = [], [], [], []
    seed = None if random_state == -1 else random_state
    random.seed(seed)

    for _ in range(max_iter):
        append = False
        i_s = random.randint(0, i_h - p_h)
        j_s = random.randint(0, i_w - p_w)
        center_i = int(i_s + (p_h / 2))
        center_j = int(j_s + (p_w / 2))

        patch = scene_dinsar[:, i_s:i_s + p_h, j_s:j_s + p_w]
        mask_patch = scene_mask[:, i_s:i_s + p_h, j_s:j_s + p_w]

        # CONDITION 1: vogliamo patch che NON CONTENGANO GL
        if float(mask_patch.sum()) == 0 and np.sum(np.abs(patch)) > 0:
            append = True
        else:
            append = False

        # se non è appendabile per mask, salta i controlli su seen
        if not append:
            continue

        # CONDITION 2: controllo 'seen' — accetta se (poco visto) OR (sufficiente % "freschi")
        # calcola box 32x32 intorno al centro con protezione bordi
        i0 = max(0, center_i - 15)
        i1 = min(seen.shape[1], center_i + 16)  # exclusive upper bound in slicing
        j0 = max(0, center_j - 15)
        j1 = min(seen.shape[2], center_j + 16)

        box_sum = float(seen[:, i0:i1, j0:j1].sum())

        cond1 = box_sum <= 200
        cond2 = check_seen_percentage(seen[:, i_s:i_s + p_h, j_s:j_s + p_w],
                                          seen_threshold=seen_threshold,
                                          perc_threshold=perc_threshold)

        # cond2 switch
        cond2 = False

        # Se nessuna delle due condizioni è vera, scarta
        if not (cond1 or cond2):
            append = False

        if append:
            assert mask_patch.size == size, "Something wrong to be checked."
            seen[:, i_s:i_s + p_h, j_s:j_s + p_w] += 1
            patches.append(patch)
            masks.append(mask_patch)
            idx_x.append(i_s)
            idx_y.append(j_s)

        if len(patches) == 10:
            break

    return patches, masks, seen, idx_x, idx_y
