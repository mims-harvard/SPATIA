import argparse
import os

import lmdb
from tqdm.auto import tqdm


def merge_lmdbs(
    source_lmdb_path, target_lmdb_path, map_size=1024**4, check_duplicate_keys=True
):
    if not os.path.exists(source_lmdb_path):
        print(f"Error: Source LMDB file '{source_lmdb_path}' not found.")
        return

    target_lmdb_dir = os.path.dirname(target_lmdb_path)
    if target_lmdb_dir and not os.path.exists(target_lmdb_dir):
        os.makedirs(target_lmdb_dir)
        print(f"Created directory for target LMDB: {target_lmdb_dir}")

    print(f"Opening target LMDB: {target_lmdb_path}")
    target_env = lmdb.open(target_lmdb_path, map_size=map_size, subdir=False, lock=True)

    print(f"Opening source LMDB: {source_lmdb_path}")
    source_env = lmdb.open(
        source_lmdb_path, map_size=map_size, subdir=False, lock=False, readonly=True
    )

    print(f"Starting merge from '{source_lmdb_path}' to '{target_lmdb_path}'...")
    with target_env.begin(write=True) as txn_target:
        with source_env.begin() as txn_source:
            cursor = txn_source.cursor()
            for key, value in tqdm(
                cursor,
                total=txn_source.stat()["entries"],
                desc=f"Merging {os.path.basename(source_lmdb_path)}",
            ):
                if check_duplicate_keys:
                    if txn_target.get(key) is not None:
                        error_msg = (
                            f"Error: Duplicate key '{key.decode()}' found in target LMDB "
                            f"'{target_lmdb_path}' while merging from '{source_lmdb_path}'. "
                            "Aborting."
                        )
                        print(error_msg)
                        raise ValueError(error_msg)
                txn_target.put(key, value)

    source_env.close()
    target_env.close()
    print("Merge complete.")
    print(f"Target LMDB '{target_lmdb_path}' updated.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Merge key-value pairs from a source LMDB file into a target LMDB file."
    )
    parser.add_argument("source_db", type=str, help="Path to the source LMDB file.")
    parser.add_argument("target_db", type=str, help="Path to the target LMDB file.")
    parser.add_argument(
        "--map_size",
        type=int,
        default=(1024**4) * 4,
        help="Maximum size the LMDB database can grow to. Defaults to 4TB.",
    )
    parser.add_argument(
        "--check_duplicate_keys",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable or disable checking for duplicate keys (default: enabled). Use --no-check_duplicate_keys to disable.",
    )

    args = parser.parse_args()

    try:
        merge_lmdbs(
            args.source_db,
            args.target_db,
            map_size=args.map_size,
            check_duplicate_keys=False,
        )
    except ValueError as e:
        exit(1)
