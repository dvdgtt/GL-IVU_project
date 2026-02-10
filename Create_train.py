import os
import argparse
import math
import pandas as pd
import numpy as np
import xarray as xr
import rioxarray
import rasterio
from rasterio.features import rasterize
import geopandas as gpd
import shapely
import matplotlib.pyplot as plt
from utils import get_phase_and_magnitude, flow_train_dataset, flow_noisy_train_dataset

parser = argparse.ArgumentParser()
parser.add_argument('--save', type=int, default=0, help="Save train dataset or not.")
parser.add_argument('--random_state', type=int, default=42, help="Random or deterministic. Use -1 for randomness.")
parser.add_argument('--path_train_data', type=str, default=r"/media/user/data_nvme/groundice/data/train/DInSAR/", help="Path to data")
parser.add_argument('--path_train_out', type=str, default=r"/media/user/data_nvme/groundice/data/train/train/128train/", help="Path to where data to be saved")
parser.add_argument('--patch_size', type=int, default=128, help="Size of images")
parser.add_argument('--all_touched', type=int, default=1, help="Burn all touching pixels or use Bresenham line algorithm")
args = parser.parse_args()

path_patches_root = args.path_train_out

# cartella per i patch della specifica dimensione (es. .../patches-train/128)
path_size_dir = os.path.join(path_patches_root, str(args.patch_size))

# cartelle interne: train128 e noise128
train_patches_dir = os.path.join(path_size_dir, f"train{args.patch_size}")
noise_patches_dir = os.path.join(path_size_dir, f"noise{args.patch_size}")

# creazione sicura delle cartelle
os.makedirs(train_patches_dir, exist_ok=True)
os.makedirs(noise_patches_dir, exist_ok=True)

# Import .gpkg
data_grounding_line_geometries = gpd.read_file(f"{args.path_train_data}/all_manual_segm_gl.gpkg")
# Remove non valid geometies: they should be 44 of them. Valid: 2524 -> 2480
data_grounding_line_geometries = data_grounding_line_geometries[~data_grounding_line_geometries.geometry.isna()]
#print(data_grounding_line_geometries)

num_training_scenes = len(os.listdir(f'{args.path_train_data}/DInSAR'))

# loop over the tif scenes
for n, file_name_tif in enumerate(os.listdir(f'{args.path_train_data}/DInSAR')):

    # get the grounding data for each tif scene
    data_gl_scene = data_grounding_line_geometries[data_grounding_line_geometries['source_file'] == file_name_tif.replace(".tif", ".shp")]
    #print(n, file_name_tif, len(data_gl_scene))

    # for some reason some scenes do not have grounding lines in the dataset. Let's skip those
    if len(data_gl_scene) == 0:
        print(f"{n} File {file_name_tif} has no grounding lines in dataset and will be skipped.")
        continue

    # open the tif scene
    tif_dinsar_re_im = rioxarray.open_rasterio(f'{args.path_train_data}/DInSAR/{file_name_tif}')
    phase, magnitude = get_phase_and_magnitude(tif_dinsar_re_im)

    # costruiamo una versione fase+coerenza (2 bande)
    tif_dinsar_phi_mag = tif_dinsar_re_im.copy(deep=True)
    tif_dinsar_phi_mag.values[0, :, :] = phase
    tif_dinsar_phi_mag.values[1, :, :] = magnitude

    # create an xarray for burning in grounding line pixels. We initialize it with zeros.
    tif_gl_mask = xr.DataArray(
        np.zeros((1, tif_dinsar_re_im.shape[1], tif_dinsar_re_im.shape[2]), dtype="uint8"),
        dims=tif_dinsar_re_im.dims,
        coords={
            "band": ["mask"],
            "y": tif_dinsar_re_im.coords["y"],
            "x": tif_dinsar_re_im.coords["x"],
        },
    )

    # Copy over rioxarray geospatial attrs
    tif_gl_mask.rio.write_crs(tif_dinsar_re_im.rio.crs, inplace=True)
    tif_gl_mask.rio.write_transform(tif_dinsar_re_im.rio.transform(), inplace=True)

    # check the projection
    assert data_gl_scene.crs == tif_gl_mask.rio.crs, "Projection not as expected: EPSG:3031"

    # Burn in geometries
    mask_array = rasterize(
        ((geom, 1) for geom in data_gl_scene.geometry),
        out_shape=(tif_gl_mask.shape[1], tif_gl_mask.shape[2]),
        transform=tif_gl_mask.rio.transform(),
        all_touched=args.all_touched,
        fill=0,
        dtype='uint8'
    )

    # Plug in the burned mask
    tif_gl_mask.values[0] = mask_array

    # At this point we have the two xarrays
    # tif_dinsar_phi_mag is the DInSAR scene in fase+coerenza
    # tif_gl_mask is the mask

    plot = False
    if plot:
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(13, 4))
        data_gl_scene.plot(ax=ax1, ec='black', fc='none', lw=2)
        data_gl_scene.plot(ax=ax2, ec='red', fc='none', lw=2)
        data_gl_scene.plot(ax=ax3, ec='red', fc='none', lw=2, alpha=0.5)
        im1 = tif_dinsar_phi_mag.sel(band=1).plot(ax=ax1, cmap='hsv', vmin=-np.pi, vmax=np.pi, add_colorbar=False)
        im2 = tif_dinsar_phi_mag.sel(band=2).plot(ax=ax2, cmap='gray', add_colorbar=False)
        im3 = tif_gl_mask.plot(ax=ax3, cmap='gray', add_colorbar=False)
        for ax in (ax1, ax2, ax3):
            ax.set_aspect('equal')
        ax1.set_title("Phase")
        ax2.set_title("Coherence")
        ax2.set_title("Mask")

        # Add colorbars below each plot
        cbar1 = plt.colorbar(im1, ax=ax1, orientation='horizontal', pad=0.1, fraction=0.046)
        cbar1.set_label("Phase [rad]")

        cbar2 = plt.colorbar(im2, ax=ax2, orientation='horizontal', pad=0.1, fraction=0.046)
        cbar2.set_label("Coherence")

        cbar3 = plt.colorbar(im3, ax=ax3, orientation='horizontal', pad=0.1, fraction=0.046)
        cbar3.set_label("Mask")

        plt.tight_layout()
        plt.show()

    # >>> QUI passiamo fase+coerenza al generatore di patch <<<
    seen, num_images = flow_train_dataset(
        scene_dinsar=tif_dinsar_phi_mag,
        scene_mask=tif_gl_mask,
        scene_ground_lines=data_gl_scene,
        scene_name=file_name_tif.replace(".tif", ""),
        patch_size=args.patch_size,
        random_state=args.random_state,
        path_out=args.path_train_out,
        save=args.save
    )

    print(f"From scene {n}/{num_training_scenes} we have produced {num_images} images of shape {args.patch_size}x{args.patch_size}.")

    seen_noise, num_noisy_images = flow_noisy_train_dataset(
        scene_dinsar=tif_dinsar_phi_mag,
        scene_mask=tif_gl_mask,
        scene_ground_lines=data_gl_scene,
        scene_name=file_name_tif.replace(".tif", ""),
        patch_size=args.patch_size,
        random_state=args.random_state,
        path_out=args.path_train_out,
        save=args.save
    )

    print(f"From scene {n}/{num_training_scenes} we have produced {num_noisy_images} noise images of shape {args.patch_size}x{args.patch_size}.")