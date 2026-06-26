
import argparse
import os
from glob import glob

import imageio
import lmdb
import numpy as np
import pandas as pd
import tifffile
from filelock import FileLock
from skimage.transform import resize
from tqdm.auto import tqdm


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Crop cell images from Xenium datasets centered on individual cells"
    )
    parser.add_argument(
        "--output-lmdb",
        type=str,
        default="dataset/lmdb/cell_crops",
        help="Output LMDB folder path",
    )
    parser.add_argument(
        "--crop-size",
        type=int,
        default=None,
        help="Size of the crop window around each cell. If not provided, the smallest square that contains the cell will be used.",
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
        help="Process only the specified dataset instead of all datasets",
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
    cache_file = os.path.join(cache_dir, f"{dataset_name}_resample_norm.npy")
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
        image = tif.series[0].levels[0].asarray().max(axis=0)

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


def get_cell_crop_coordinates(cell, image_height, image_width, crop_radius=None):
    cell_x = int(cell["y_centroid"] / 0.2125)
    cell_y = int(cell["x_centroid"] / 0.2125)

    if crop_radius is not None:
        start_x = max(0, cell_x - crop_radius)
        start_y = max(0, cell_y - crop_radius)
        end_x = min(image_height, cell_x + crop_radius)
        end_y = min(image_width, cell_y + crop_radius)
        crop_size = crop_radius * 2
    else:
        if all(
            col in cell.index for col in ["x_min", "x_max", "y_min", "y_max"]
        ) and all(
            not pd.isna(cell[col]) for col in ["x_min", "x_max", "y_min", "y_max"]
        ):
            x_min = max(0, int(cell["y_min"] / 0.2125))
            x_max = min(image_height, int(cell["y_max"] / 0.2125))
            y_min = max(0, int(cell["x_min"] / 0.2125))
            y_max = min(image_width, int(cell["x_max"] / 0.2125))

            half_side = max(
                abs(x_max - cell_x),
                abs(x_min - cell_x),
                abs(y_max - cell_y),
                abs(y_min - cell_y),
            )
            half_side = max(half_side, 4)

            start_x = max(0, cell_x - half_side)
            start_y = max(0, cell_y - half_side)
            end_x = min(image_height, cell_x + half_side)
            end_y = min(image_width, cell_y + half_side)
            crop_size = half_side * 2
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
        pad_x_end = pad_x_start + (end_x - start_x)
        pad_y_end = pad_y_start + (end_y - start_y)
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


def draw_cell_boundary(
    image, cell, start_x, start_y, scale_factor, output_size, center_x, center_y
):
    if all(col in cell.index for col in ["x_min", "x_max", "y_min", "y_max"]) and all(
        not pd.isna(cell[col]) for col in ["x_min", "x_max", "y_min", "y_max"]
    ):
        local_x_min = int((int(cell["y_min"] / 0.2125) - start_x) * scale_factor)
        local_x_max = int((int(cell["y_max"] / 0.2125) - start_x) * scale_factor)
        local_y_min = int((int(cell["x_min"] / 0.2125) - start_y) * scale_factor)
        local_y_max = int((int(cell["x_max"] / 0.2125) - start_y) * scale_factor)

        if (
            local_x_max >= 0
            and local_y_max >= 0
            and local_x_min < output_size
            and local_y_min < output_size
        ):
            local_x_min = max(0, min(local_x_min, output_size - 1))
            local_x_max = max(0, min(local_x_max, output_size - 1))
            local_y_min = max(0, min(local_y_min, output_size - 1))
            local_y_max = max(0, min(local_y_max, output_size - 1))

            image = draw_bounding_box(
                image,
                local_x_min,
                local_y_min,
                local_x_max,
                local_y_max,
                color=[0, 0, 255],
                thickness=1,
            )
    else:
        box_size = max(2, int(5 * scale_factor))
        local_x_min = max(0, center_x - box_size)
        local_y_min = max(0, center_y - box_size)
        local_x_max = min(output_size - 1, center_x + box_size)
        local_y_max = min(output_size - 1, center_y + box_size)

        image = draw_bounding_box(
            image,
            local_x_min,
            local_y_min,
            local_x_max,
            local_y_max,
            color=[0, 0, 255],
            thickness=1,
        )

    return image


def create_example_image(
    crop_img,
    cell,
    start_x,
    start_y,
    crop_radius,
    output_size,
    crop_size,
    draw_boundaries=True,
):
    marked_img = np.stack([crop_img.copy()] * 3, axis=-1)

    cell_x = int(cell["y_centroid"] / 0.2125)
    cell_y = int(cell["x_centroid"] / 0.2125)

    center_x = crop_radius
    center_y = crop_radius
    if cell_x < crop_radius:
        center_x = cell_x
    if cell_y < crop_radius:
        center_y = cell_y

    scale_factor = output_size / crop_size if output_size != crop_size else 1.0

    center_x = int(center_x * scale_factor)
    center_y = int(center_y * scale_factor)

    marked_img = mark_cell_center(marked_img, center_x, center_y, output_size)

    if draw_boundaries:
        marked_img = draw_cell_boundary(
            marked_img,
            cell,
            start_x,
            start_y,
            scale_factor,
            output_size,
            center_x,
            center_y,
        )

    cell_type = cell.get("cell_type", "unknown")
    if pd.isna(cell_type):
        cell_type = "unknown"

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

    if "zip" in dataset_name:
        return

    mipurl = f"dataset/download/{dataset_name}/morphology.ome.tif"
    if not os.path.exists(mipurl):
        print(f"Missing {mipurl}")
        return

    map_size = 1024**4
    env = lmdb.open(lmdb_path, map_size=map_size, subdir=False, lock=True)

    print(f"Processing {mipurl}")

    resample_norm = load_and_normalize_image(mipurl, dataset_name, args.cache)

    cells = pd.read_parquet(f"dataset/download/{dataset_name}/cells.parquet")
    print(f"Loaded {len(cells)} cells")

    if args.max_cells and len(cells) > args.max_cells:
        cells = cells.sample(args.max_cells, random_state=42)
        print(f"Sampled {len(cells)} cells for processing")

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
    else:
        print("Skipping example image saving")

    assert (
        max(cells["y_centroid"]) < height
    ), f"y_centroid exceeds image height: {max(cells['y_centroid'])} >= {height}"
    assert (
        max(cells["x_centroid"]) < width
    ), f"x_centroid exceeds image width: {max(cells['x_centroid'])} >= {width}"

    crop_radius = None if args.crop_size is None else args.crop_size // 2

    with env.begin(write=True) as txn:
        for idx, cell in tqdm(
            cells.iterrows(),
            total=len(cells),
            desc=f"Processing cells in {dataset_name}",
        ):
            cell_x, cell_y, start_x, start_y, end_x, end_y, crop_size = (
                get_cell_crop_coordinates(cell, height, width, crop_radius)
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

            cell_id = f"{cell.get('cell_id', idx)}"

            if args.save_examples and len(example_images) < args.max_examples:
                effective_crop_radius = (
                    crop_radius if crop_radius is not None else crop_size // 2
                )
                marked_img, cell_type = create_example_image(
                    crop_img,
                    cell,
                    start_x,
                    start_y,
                    effective_crop_radius,
                    args.output_size,
                    crop_size,
                    args.draw_cell_boundaries,
                )
                example_images.append((cell_id, marked_img, cell_type))

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
        dataset_names = [f"dataset/download/{args.dataset_name}"]
        if not os.path.exists(dataset_names[0]):
            print(f"指定的数据集不存在: {dataset_names[0]}")
            exit(1)
    else:
        dataset_names = glob("dataset/download/Xenium*")

    for dataset_path in tqdm(dataset_names, total=len(dataset_names)):
        process_dataset(dataset_path, args)

    print("处理完成!")


if __name__ == "__main__":
    main()
