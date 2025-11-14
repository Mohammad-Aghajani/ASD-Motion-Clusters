"""CLI wrapper for utils_cleaned.ingest_new_data."""

import argparse

from utils_cleaned import ingest_new_data


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest raw motion files into data/Actor NN folders."
    )
    parser.add_argument('--src', required=True, help='Folder containing new .txt files')
    parser.add_argument('--data-root', default='data', help='Destination data folder (default: data)')
    args = parser.parse_args()

    mapping = ingest_new_data(args.src, data_root=args.data_root)
    if mapping:
        print("\nActor mapping (source -> new ID):")
        for src_key, new_id in mapping.items():
            print(f"  {src_key} -> {new_id}")


if __name__ == '__main__':
    main()
