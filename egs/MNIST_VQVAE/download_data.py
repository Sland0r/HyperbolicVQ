"""Download MNIST dataset to the project dataset directory."""
import argparse
from torchvision import datasets


def main():
    parser = argparse.ArgumentParser(description="Download MNIST dataset")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/home/acolombo/VAEs/dataset/MNIST",
        help="directory to download MNIST into",
    )
    args = parser.parse_args()

    print(f"Downloading MNIST to: {args.data_dir}")
    datasets.MNIST(root=args.data_dir, train=True, download=True)
    datasets.MNIST(root=args.data_dir, train=False, download=True)
    print("Done. Train and test splits downloaded.")


if __name__ == "__main__":
    main()
