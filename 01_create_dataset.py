import argparse
from pathlib import Path
import numpy as np
import zarr


def get_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--mode", type=str, default="train")  # train or test
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    mode = args.mode  # train or test
    root_dir = Path(__file__).parent.joinpath("input")
    output_root = Path(__file__).parent.joinpath("output")
    output_dir = output_root.joinpath(f"{mode}_imgs")
    output_dir.mkdir(exist_ok=True, parents=True)

    for exp_dir in root_dir.joinpath(mode, "static", "ExperimentRuns").iterdir():
        if not exp_dir.is_dir():
            continue

        print(exp_dir)
        zarr_path = exp_dir.joinpath("VoxelSpacing10.000", "denoised.zarr")
        zarr_file = zarr.open(str(zarr_path))
        tomogram = zarr_file["0"][:]
        output_path = output_dir.joinpath(f"{exp_dir.stem}.npy")
        np.save(str(output_path), tomogram)


if __name__ == '__main__':
    main()
