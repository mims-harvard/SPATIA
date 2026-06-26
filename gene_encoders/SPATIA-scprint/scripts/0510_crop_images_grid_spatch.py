
import argparse
import os
from collections import Counter
from glob import glob

import anndata as ad
import imageio
import lmdb
import numpy as np
import pandas as pd
import scanpy as sc
import tifffile
from filelock import FileLock
from skimage.transform import resize
from tqdm.auto import tqdm


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Crop grid images from spatch datasets"
    )
    parser.add_argument(
        "--output-lmdb",
        type=str,
        default="dataset/lmdb/grid_spatch",
        help="Output LMDB folder path",
    )
    parser.add_argument(
        "--crop-size", type=int, default=256, help="Size of the grid for cropping"
    )
    parser.add_argument(
        "--output-size", type=int, default=256, help="Size of the output images"
    )
    parser.add_argument(
        "--no-save-examples",
        action="store_true",
        help="Do not save example images for visual inspection",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only process enough to save example images, then exit",
    )
    parser.add_argument(
        "--cache",
        type=str,
        default="dataset/cache",
        help="Cache folder for preprocessed images",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default=None,
        help="Process only the specified dataset instead of all datasets (COAD, HCC, OV)",
    )
    parser.add_argument(
        "--draw-cell-boundaries",
        action="store_true",
        help="Draw blue bounding boxes for cell boundaries in example images",
    )
    parser.add_argument(
        "--image-file",
        type=str,
        default="DAPI.tif",
        help="Image file to use for cropping (DAPI.tif, CODEX.tif, or HE.tif)",
    )
    args = parser.parse_args()

    args.save_examples = not args.no_save_examples
    args.draw_cell_boundaries = True

    return args


def draw_bounding_box(
    image, x_min, y_min, x_max, y_max, color=[0, 0, 255], thickness=1
):
    x_min, y_min, x_max, y_max = int(x_min), int(y_min), int(x_max), int(y_max)

    image[x_min : x_min + thickness, y_min : y_max + 1] = color
    image[x_max : x_max + thickness, y_min : y_max + 1] = color

    image[x_min : x_max + 1, y_min : y_min + thickness] = color
    image[x_min : x_max + 1, y_max : y_max + thickness] = color

    return image


def load_and_normalize_image(mipurl, dataset_name, cache_dir):
    cache_file = os.path.join(cache_dir, f"spatch_{dataset_name}_resample_norm.npy")
    cache_lock = cache_file + ".lock"

    if os.path.exists(cache_file):
        with FileLock(cache_lock):
            print(f"Loading cached normalized image from {cache_file}")
            resample_norm = np.load(cache_file)
    else:
        tif = tifffile.TiffFile(mipurl)

        for tag in tif.pages[0].tags.values():
            if tag.name == "ImageDescription":
                print(tag.name + ":", tag.value)

        print("tif.series[0].levels[0].shape", tif.series[0].levels[0].shape)
        image = tif.series[0].levels[0].asarray()

        if len(image.shape) > 2:
            print(f"多维图像: {image.shape}, 取最大投影")
            image = image.max(axis=0)

        print(f"Image shape: {image.shape}")
        resample = image

        resample_norm = (
            (resample - resample.min()) / (resample.max() - resample.min()) * 255
        )
        resample_norm = resample_norm.astype("uint8")

        with FileLock(cache_lock):
            print(f"Caching normalized image to {cache_file}")
            np.save(cache_file, resample_norm)

    return resample_norm


def calculate_grid_dimensions(image_height, image_width, grid_size):
    n_rows = image_height // grid_size + (1 if image_height % grid_size > 0 else 0)
    n_cols = image_width // grid_size + (1 if image_width % grid_size > 0 else 0)

    return n_rows, n_cols


def assign_cells_to_grids(
    cell_positions, height, width, grid_size, n_rows, n_cols, resolution
):
    cell_pos_list = []

    for i, pos in enumerate(cell_positions):
        x = int(pos[1] / resolution)
        y = int(pos[0] / resolution)

        if 0 <= x < height and 0 <= y < width:
            grid_row = x // grid_size
            grid_col = y // grid_size

            if grid_row < n_rows and grid_col < n_cols:
                cell_pos_list.append((i, grid_row, grid_col))

    grid_cells = {}
    for cell_idx, grid_row, grid_col in cell_pos_list:
        grid_key = (grid_row, grid_col)
        if grid_key not in grid_cells:
            grid_cells[grid_key] = []
        grid_cells[grid_key].append(cell_idx)

    return cell_pos_list, grid_cells


def crop_and_pad_grid(
    image, start_row, start_col, grid_size, output_size, image_height, image_width
):
    end_row = min(start_row + grid_size, image_height)
    end_col = min(start_col + grid_size, image_width)

    crop_img = image[start_row:end_row, start_col:end_col]

    if crop_img.shape != (grid_size, grid_size):
        padded = np.zeros((grid_size, grid_size), dtype=np.uint8)
        padded[: crop_img.shape[0], : crop_img.shape[1]] = crop_img
        crop_img = padded

    if output_size != grid_size:
        crop_img = resize(
            crop_img,
            (output_size, output_size),
            preserve_range=True,
            anti_aliasing=True,
        ).astype(np.uint8)

    return crop_img


def mark_cell_centers(
    marked_img,
    cell_indices,
    cell_positions,
    start_row,
    start_col,
    scale_factor,
    resolution,
):
    for idx in cell_indices:
        pos = cell_positions[idx]

        local_x = int(pos[1] / resolution) - start_row
        local_y = int(pos[0] / resolution) - start_col

        local_x = int(local_x * scale_factor)
        local_y = int(local_y * scale_factor)

        if 0 <= local_x < marked_img.shape[0] and 0 <= local_y < marked_img.shape[1]:
            x_min = max(0, local_x - 2)
            x_max = min(marked_img.shape[0], local_x + 2)
            y_min = max(0, local_y - 2)
            y_max = min(marked_img.shape[1], local_y + 2)
            marked_img[x_min:x_max, y_min:y_max] = [255, 0, 0]

    return marked_img


def create_example_image(
    crop_img,
    cell_indices,
    cell_positions,
    start_row,
    start_col,
    grid_size,
    output_size,
    resolution,
):
    marked_img = np.stack([crop_img.copy()] * 3, axis=-1)

    scale_factor = output_size / grid_size if output_size != grid_size else 1.0

    marked_img = mark_cell_centers(
        marked_img,
        cell_indices,
        cell_positions,
        start_row,
        start_col,
        scale_factor,
        resolution,
    )

    marked_img = draw_bounding_box(
        marked_img,
        0,
        0,
        marked_img.shape[0] - 1,
        marked_img.shape[1] - 1,
        color=[0, 255, 0],
        thickness=2,
    )

    return marked_img


def save_to_lmdb(txn, dataset_name, grid_id, crop_img):
    data_key = f"{dataset_name}/{grid_id}".encode("utf-8")
    raw_bytes = crop_img.tobytes()
    txn.put(data_key, raw_bytes)


def compute_grid_metadata(cell_subset, grid_metadata):
    for col in cell_subset.obs.columns:
        if col not in grid_metadata:
            try:
                values = cell_subset.obs[col].values
                most_common_value = Counter(values[~pd.isna(values)]).most_common(1)
                if most_common_value:
                    grid_metadata[col] = most_common_value[0][0]
                else:
                    grid_metadata[col] = None
            except:
                if pd.api.types.is_numeric_dtype(cell_subset.obs[col]):
                    grid_metadata[col] = cell_subset.obs[col].mean()
                else:
                    grid_metadata[col] = (
                        cell_subset.obs[col].iloc[0]
                        if not cell_subset.obs[col].empty
                        else None
                    )
    return grid_metadata


def save_example_images(example_images, example_folder, dataset_name):
    print(f"Saving {len(example_images)} example images to {example_folder}")
    for idx, (grid_id, img, num_cells) in enumerate(example_images):
        imageio.imwrite(
            os.path.join(
                example_folder,
                f"{dataset_name}_{grid_id}_{idx}_{num_cells}cells.png",
            ),
            img,
        )


def create_grid_adata(all_grid_expr, all_grid_metadata, adata_orig, h5ad_path):
    if all_grid_expr and all_grid_metadata:
        print(f"Creating h5ad file at {h5ad_path}")

        grid_expr_matrix = np.vstack(all_grid_expr)

        adata = ad.AnnData(X=grid_expr_matrix)

        adata.obs = pd.DataFrame(all_grid_metadata).set_index("cell_id")

        adata.var = adata_orig.var.copy()

        adata.write(h5ad_path)
        print(f"Saved h5ad file with {adata.shape[0]} grids and {adata.shape[1]} genes")


def process_dataset(dataset_path, args):
    dataset_name = os.path.basename(dataset_path)
    lmdb_path = f"{args.output_lmdb}/{dataset_name}.lmdb"
    h5ad_path = f"{args.output_lmdb}/{dataset_name}.h5ad"
    done_marker = f"{args.output_lmdb}/{dataset_name}.DONE"

    if os.path.exists(done_marker):
        print(f"Skip {dataset_name} - found marker file {done_marker}")
        return

    image_file = os.path.join(dataset_path, args.image_file)
    adata_file = os.path.join(dataset_path, "adata.h5ad")

    if not os.path.exists(image_file):
        print(f"Missing image file: {image_file}")
        return

    if not os.path.exists(adata_file):
        print(f"Missing AnnData file: {adata_file}")
        return

    map_size = 1024**4
    env = lmdb.open(lmdb_path, map_size=map_size, subdir=False, lock=True)

    print(f"Processing {dataset_name}")

    print(f"Loading image from {image_file}")
    resample_norm = load_and_normalize_image(image_file, dataset_name, args.cache)

    print(f"Loading cell data from {adata_file}")
    adata = sc.read_h5ad(adata_file)

    if args.image_file == "DAPI.tif":
        resolution = adata.uns["DAPI resolution"]
    elif args.image_file == "CODEX.tif":
        resolution = adata.uns["CODEX resolution"]
    elif args.image_file == "HE.tif":
        resolution = adata.uns["H&E resolution"]
    else:
        resolution = adata.uns["DAPI resolution"]

    print(f"Using resolution: {resolution}")

    cell_positions = adata.obsm["spatial"]

    height, width = resample_norm.shape
    print(f"Image size: {height}x{width}")

    grid_size = args.crop_size
    n_rows, n_cols = calculate_grid_dimensions(height, width, grid_size)
    print(f"Grid dimensions: {n_rows}x{n_cols}, total grids: {n_rows * n_cols}")

    cell_pos_list, grid_cells = assign_cells_to_grids(
        cell_positions, height, width, grid_size, n_rows, n_cols, resolution
    )

    print(f"Assigned {len(cell_pos_list)} cells to {len(grid_cells)} grids")

    example_images = []
    example_folder = None
    if args.save_examples:
        example_folder = os.path.join(
            os.path.dirname(lmdb_path), os.path.basename(lmdb_path).split(".")[0]
        )
        if not os.path.exists(example_folder):
            os.makedirs(example_folder)
        print(f"Saving example images to {example_folder}")

    all_grid_expr = []
    all_grid_metadata = []

    with env.begin(write=True) as txn:
        grid_counter = 0
        for grid_key in tqdm(grid_cells.keys(), desc="Processing grids with cells"):
            grid_row, grid_col = grid_key
            cell_indices = grid_cells[grid_key]

            start_row = grid_row * grid_size
            start_col = grid_col * grid_size

            crop_img = crop_and_pad_grid(
                resample_norm,
                start_row,
                start_col,
                grid_size,
                args.output_size,
                height,
                width,
            )

            grid_id = f"grid_{grid_row}_{grid_col}_{grid_size}_{args.output_size}"

            if args.save_examples and len(example_images) < 100:
                marked_img = create_example_image(
                    crop_img,
                    cell_indices,
                    cell_positions,
                    start_row,
                    start_col,
                    grid_size,
                    args.output_size,
                    resolution,
                )

                example_images.append((grid_id, marked_img, len(cell_indices)))

            save_to_lmdb(txn, dataset_name, grid_id, crop_img)

            cell_subset = adata[cell_indices]

            grid_expr = cell_subset.X.sum(axis=0)

            all_grid_expr.append(grid_expr)

            grid_metadata = {
                "cell_id": grid_id,
                "grid_row": grid_row,
                "grid_col": grid_col,
                "num_cells": len(cell_indices),
                "x_min": start_row,
                "x_max": start_row + grid_size,
                "y_min": start_col,
                "y_max": start_col + grid_size,
                "cell_ids": ",".join(cell_subset.obs.index),
            }

            grid_metadata = compute_grid_metadata(cell_subset, grid_metadata)
            all_grid_metadata.append(grid_metadata)

            grid_counter += 1

            if grid_counter % 1000 == 0:
                print(f"Processed {grid_counter} grids with cells")

            if args.dry_run and len(example_images) >= 100:
                break

    if args.save_examples and example_images:
        save_example_images(example_images, example_folder, dataset_name)

    if args.dry_run:
        print("Dry run completed, exiting without creating h5ad file")
        env.close()
        return

    if all_grid_expr and all_grid_metadata:
        create_grid_adata(all_grid_expr, all_grid_metadata, adata, h5ad_path)

    if not args.dry_run:
        with open(done_marker, "w") as f:
            f.write(f"Processed on {pd.Timestamp.now()}")
        print(f"Created marker file {done_marker}")

    env.close()


def main():
    args = parse_arguments()


    if not os.path.exists(args.cache):
        os.makedirs(args.cache)

    if not os.path.exists(args.output_lmdb):
        os.makedirs(args.output_lmdb, exist_ok=True)

    if args.dataset_name:
        dataset_path = f"dataset/download/all/Xenium/{args.dataset_name}"
        if not os.path.exists(dataset_path):
            print(f"指定的数据集不存在: {dataset_path}")
            exit(1)
        dataset_paths = [dataset_path]
    else:
        xenium_dir = "dataset/download/all/Xenium"
        if not os.path.exists(xenium_dir):
            print(f"Xenium目录不存在: {xenium_dir}")
            exit(1)

        dataset_paths = [
            os.path.join(xenium_dir, d)
            for d in os.listdir(xenium_dir)
            if os.path.isdir(os.path.join(xenium_dir, d))
        ]

        if len(dataset_paths) == 0:
            print(f"在 {xenium_dir} 中找不到数据集")
            exit(1)

    for dataset_path in tqdm(dataset_paths, total=len(dataset_paths)):
        process_dataset(dataset_path, args)

    print("处理完成!")


if __name__ == "__main__":
    main()
