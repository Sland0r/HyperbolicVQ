"""Download EMNIST dataset using torchvision."""
import argparse
from torchvision import datasets


def main():
    parser = argparse.ArgumentParser(description="Download EMNIST dataset")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/home/acolombo/VAEs/dataset/EMNIST",
        help="directory to download EMNIST into",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="byclass",
        choices=["byclass", "bymerge", "balanced", "letters", "digits", "mnist"],
        help="which EMNIST split to download (default: byclass)",
    )
    args = parser.parse_args()

    print(f"Downloading EMNIST (split={args.split}) to: {args.data_dir}")
    datasets.EMNIST(root=args.data_dir, split=args.split, train=True, download=True)
    datasets.EMNIST(root=args.data_dir, split=args.split, train=False, download=True)
    print("Done. Train and test splits downloaded.")


if __name__ == "__main__":
    main()
