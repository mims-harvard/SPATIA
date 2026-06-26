
import argparse
import os
from glob import glob

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
        description="Crop cell images from spatch datasets centered on individual cells"
    )
    parser.add_argument(
        "--output-lmdb",
        type=str,
        default="dataset/lmdb/cell_crops_spatch",
        help="Output LMDB folder path",
    )
    parser.add_argument(
        "--crop-size",
        type=int,
        default=None,
        help="Size of the crop window around each cell. If not provided, a default size will be used.",
    )
    parser.add_argument(
        "--output-size",
        type=int,
        default=256,
        help="Size of the output images",
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
        "--max-cells",
        type=int,
        default=None,
        help="Maximum number of cells to process per dataset (for testing)",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=10,
        help="Maximum number of example images to save",
    )
    parser.add_argument(
        "--image-file",
        type=str,
        default="DAPI.tif",
        help="Image file to use for cropping (DAPI.tif, CODEX.tif, or HE.tif)",
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="dataset/download/all/Xenium",
        help="Directory containing per-dataset subfolders (e.g. HCC, COAD, OV), "
        "each with adata.h5ad and the image file",
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


def get_cell_crop_coordinates(
    cell_pos, image_height, image_width, resolution, crop_radius=None
):
    cell_x = int(cell_pos[1] / resolution)
    cell_y = int(cell_pos[0] / resolution)

    if crop_radius is not None:
        start_x = max(0, cell_x - crop_radius)
        start_y = max(0, cell_y - crop_radius)
        end_x = min(image_height, cell_x + crop_radius)
        end_y = min(image_width, cell_y + crop_radius)
        crop_size = crop_radius * 2
    else:
        default_radius = 32
        start_x = max(0, cell_x - default_radius)
        start_y = max(0, cell_y - default_radius)
        end_x = min(image_height, cell_x + default_radius)
        end_y = min(image_width, cell_y + default_radius)
        crop_size = default_radius * 2

    return cell_x, cell_y, start_x, start_y, end_x, end_y, crop_size


def crop_and_pad_image(
    image, start_x, start_y, end_x, end_y, crop_size, cell_x, cell_y
):
    crop_img = image[start_x:end_x, start_y:end_y]
    crop_radius = crop_size // 2

    if crop_img.shape != (crop_size, crop_size):
        padded = np.zeros((crop_size, crop_size), dtype=np.uint8)
        pad_x_start = max(0, crop_radius - cell_x)
        pad_y_start = max(0, crop_radius - cell_y)
        actual_h, actual_w = crop_img.shape[:2]
        pad_x_end = pad_x_start + actual_h
        pad_y_end = pad_y_start + actual_w
        if actual_h > 0 and actual_w > 0:
            padded[pad_x_start:pad_x_end, pad_y_start:pad_y_end] = crop_img
        crop_img = padded

    return crop_img


def resize_image(image, output_size):
    if output_size != image.shape[0]:
        return resize(
            image,
            (output_size, output_size),
            preserve_range=True,
            anti_aliasing=True,
        ).astype(np.uint8)
    return image


def mark_cell_center(image, center_x, center_y, output_size):
    if 0 <= center_x < output_size and 0 <= center_y < output_size:
        x_min = max(0, center_x - 2)
        x_max = min(output_size, center_x + 2)
        y_min = max(0, center_y - 2)
        y_max = min(output_size, center_y + 2)
        image[x_min:x_max, y_min:y_max] = [255, 0, 0]

    return image


def create_example_image(
    crop_img,
    cell_type,
    start_x,
    start_y,
    crop_radius,
    output_size,
    crop_size,
    draw_boundaries=True,
):
    marked_img = np.stack([crop_img.copy()] * 3, axis=-1)

    center_x = crop_radius
    center_y = crop_radius

    scale_factor = output_size / crop_size if output_size != crop_size else 1.0

    center_x = int(center_x * scale_factor)
    center_y = int(center_y * scale_factor)

    marked_img = mark_cell_center(marked_img, center_x, center_y, output_size)

    if draw_boundaries:
        box_size = max(2, int(5 * scale_factor))
        local_x_min = max(0, center_x - box_size)
        local_y_min = max(0, center_y - box_size)
        local_x_max = min(output_size - 1, center_x + box_size)
        local_y_max = min(output_size - 1, center_y + box_size)

        marked_img = draw_bounding_box(
            marked_img,
            local_x_min,
            local_y_min,
            local_x_max,
            local_y_max,
            color=[0, 0, 255],
            thickness=1,
        )

    if pd.isna(cell_type):
        cell_type = "unknown"
    else:
        cell_type = str(cell_type).replace(" ", "_").replace("/", "-")

    return marked_img, cell_type


def save_to_lmdb(txn, dataset_name, cell_id, crop_img):
    data_key = f"{dataset_name}/{cell_id}".encode("utf-8")
    raw_bytes = crop_img.tobytes()
    txn.put(data_key, raw_bytes)


def save_example_images(example_images, example_folder, dataset_name):
    print(f"Saving {len(example_images)} example images to {example_folder}")
    for idx, (cell_id, img, cell_type) in enumerate(example_images):
        imageio.imwrite(
            os.path.join(
                example_folder,
                f"{dataset_name}_{cell_id}_{idx}_{cell_type}.png",
            ),
            img,
        )


def process_dataset(dataset_path, args):
    dataset_name = os.path.basename(dataset_path)
    lmdb_path = f"{args.output_lmdb}/{dataset_name}.lmdb"
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
    cell_ids = np.array(adata.obs.index)

    if "annotation" in adata.obs.columns:
        cell_types = adata.obs["annotation"].values
    else:
        cell_types = np.array(["unknown"] * len(cell_ids))

    print(f"Loaded {len(cell_ids)} cells")

    if args.max_cells and len(cell_ids) > args.max_cells:
        indices = np.random.choice(len(cell_ids), args.max_cells, replace=False)
        cell_positions = cell_positions[indices]
        cell_ids = cell_ids[indices]
        cell_types = cell_types[indices]
        print(f"Sampled {len(indices)} cells for processing")

    height, width = resample_norm.shape
    print(f"Image size: {height}x{width}")

    example_images = []
    example_folder = None
    if args.save_examples:
        example_folder = os.path.join(
            os.path.dirname(lmdb_path), os.path.basename(lmdb_path).split(".")[0]
        )
        if not os.path.exists(example_folder):
            os.makedirs(example_folder)
        print(f"Saving example images to {example_folder}")

    crop_radius = None if args.crop_size is None else args.crop_size // 2

    with env.begin(write=True) as txn:
        for i, (cell_pos, cell_id, cell_type) in tqdm(
            enumerate(zip(cell_positions, cell_ids, cell_types)),
            total=len(cell_ids),
            desc=f"Processing cells in {dataset_name}",
        ):
            cell_x, cell_y, start_x, start_y, end_x, end_y, crop_size = (
                get_cell_crop_coordinates(
                    cell_pos, height, width, resolution, crop_radius
                )
            )

            crop_img = crop_and_pad_image(
                resample_norm,
                start_x,
                start_y,
                end_x,
                end_y,
                crop_size,
                cell_x,
                cell_y,
            )

            crop_img = resize_image(crop_img, args.output_size)

            if args.save_examples and len(example_images) < args.max_examples:
                effective_crop_radius = (
                    crop_radius if crop_radius is not None else crop_size // 2
                )
                marked_img, cell_type_str = create_example_image(
                    crop_img,
                    cell_type,
                    start_x,
                    start_y,
                    effective_crop_radius,
                    args.output_size,
                    crop_size,
                    args.draw_cell_boundaries,
                )
                example_images.append((cell_id, marked_img, cell_type_str))

            save_to_lmdb(txn, dataset_name, cell_id, crop_img)

            if args.dry_run and len(example_images) >= args.max_examples:
                print("Dry run completed with enough examples, exiting loop")
                break

    if args.save_examples and example_images:
        save_example_images(example_images, example_folder, dataset_name)

    if args.dry_run:
        print("Dry run completed, exiting")
        env.close()
        return

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
        dataset_path = os.path.join(args.input_dir, args.dataset_name)
        if not os.path.exists(dataset_path):
            print(f"指定的数据集不存在: {dataset_path}")
            exit(1)
        dataset_paths = [dataset_path]
    else:
        xenium_dir = args.input_dir
        if not os.path.exists(xenium_dir):
            print(f"输入目录不存在: {xenium_dir}")
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
