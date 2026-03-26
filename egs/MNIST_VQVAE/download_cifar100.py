"""Download CIFAR-100 dataset using torchvision."""
import argparse
from torchvision import datasets

def main():
    parser = argparse.ArgumentParser(description="Download CIFAR-100")
    parser.add_argument("--data_dir", type=str,
                        default="/home/acolombo/VAEs/dataset/CIFAR100",
                        help="directory to download CIFAR-100 to")
    args = parser.parse_args()

    print(f"Downloading CIFAR-100 to {args.data_dir}...")
    datasets.CIFAR100(root=args.data_dir, train=True, download=True)
    datasets.CIFAR100(root=args.data_dir, train=False, download=True)
    print("Done.")

if __name__ == "__main__":
    main()
