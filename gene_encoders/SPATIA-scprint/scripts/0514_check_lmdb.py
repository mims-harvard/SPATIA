import lmdb


def key_exists_in_lmdb(db_path: str, key_to_check: str) -> bool:
    try:
        env = lmdb.open(
            db_path,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
            subdir=False,
        )
    except lmdb.Error as e:
        print(f"Error opening LMDB database at {db_path}: {e}")
        return False

    key_bytes = key_to_check.encode("utf-8")

    with env.begin() as txn:
        value = txn.get(key_bytes)
        exists = value is not None

    env.close()
    return exists


if __name__ == "__main__":

    example_db_path = "/path/to/dataset/lmdb/xenium_multiscale.lmdb"
    key_to_find = "Xenium_V1_hTonsil_reactive_follicular_hyperplasia_section_FFPE_outs/grid_0_16_1024_256"
    print(
        f"Checking if key '{key_to_find}' exists in LMDB database at '{example_db_path}'..."
    )
    exists = key_exists_in_lmdb(example_db_path, key_to_find)
    if exists:
        print(f"Key '{key_to_find}' exists in the LMDB database.")
    else:
        print(f"Key '{key_to_find}' does not exist in the LMDB database.")
