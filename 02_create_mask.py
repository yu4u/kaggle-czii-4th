import argparse
from pathlib import Path
import json
import numpy as np

from src.metric import particle_types


def get_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--mode", type=str, default="train")  # train or test
    parser.add_argument("--class_num", type=int, default=5)  # 5 or 6
    args = parser.parse_args()
    return args


def create_gaussian_patch(patch_size, sigma):
    """
    Create a 3D Gaussian patch with the specified size and standard deviation.

    :param patch_size: Size of the cubic patch (patch_size, patch_size, patch_size).
    :param sigma: Standard deviation of the Gaussian.
    :return: 3D Gaussian patch as a 3D numpy array.
    """
    center = patch_size // 2
    x = np.arange(patch_size) - center
    y = np.arange(patch_size) - center
    z = np.arange(patch_size) - center
    x, y, z = np.meshgrid(x, y, z, indexing='ij')
    gaussian = np.exp(-(x ** 2 + y ** 2 + z ** 2) / (2 * sigma ** 2))
    gaussian *= 255.0 / np.max(gaussian)
    return gaussian.astype(np.uint8)



def place_patch(volume, patch, x, y, z):
    """
    Place a 3D Gaussian patch on the volume at the specified position, taking the maximum of the existing values and the patch.

    :param volume: The base 3D volume where the patch will be placed.
    :param patch: The 3D Gaussian patch to be placed.
    :param x: X-coordinate where the patch will be centered.
    :param y: Y-coordinate where the patch will be centered.
    :param z: Z-coordinate where the patch will be centered.
    """
    patch_size = patch.shape[0]
    half_size = patch_size // 2

    # Determine the region of the volume to place the patch
    x_start = max(0, x - half_size)
    x_end = min(volume.shape[2], x + half_size)
    y_start = max(0, y - half_size)
    y_end = min(volume.shape[1], y + half_size)
    z_start = max(0, z - half_size)
    z_end = min(volume.shape[0], z + half_size)

    # Calculate the corresponding region in the patch
    patch_x_start = half_size - (x - x_start)
    patch_x_end = half_size + (x_end - x)
    patch_y_start = half_size - (y - y_start)
    patch_y_end = half_size + (y_end - y)
    patch_z_start = half_size - (z - z_start)
    patch_z_end = half_size + (z_end - z)

    # Take the maximum of the existing values and the patch
    volume[z_start:z_end, y_start:y_end, x_start:x_end] = np.maximum(
        volume[z_start:z_end, y_start:y_end, x_start:x_end],
        patch[patch_z_start:patch_z_end, patch_y_start:patch_y_end, patch_x_start:patch_x_end]
    )


def main():
    args = get_args()
    mode = args.mode  # train or test
    class_num = args.class_num
    root_dir = Path(__file__).parent.joinpath("input")
    output_root = Path(__file__).parent.joinpath("output")
    output_dir = output_root.joinpath(f"{mode}_masks")
    output_dir.mkdir(exist_ok=True, parents=True)
    json_names = [particle_type + ".json" for particle_type in particle_types[:class_num]]

    gaussian_patch = create_gaussian_patch(45, 6)

    for exp_dir in root_dir.joinpath(mode, "overlay", "ExperimentRuns").iterdir():
        if not exp_dir.is_dir():
            continue

        print(exp_dir)
        json_dir = exp_dir.joinpath("Picks")
        output_mask = np.zeros((184, 630, 630, class_num), dtype=np.uint8)
        d, h, w, _ = output_mask.shape

        for i, json_name in enumerate(json_names):
            json_path = json_dir.joinpath(json_name)

            with open(json_path) as f:
                picks = json.load(f)

                # [10.012444196428572, 10.012444196428572, 10.012444537618887]

                for point in picks["points"]:
                    x = int(point["location"]["x"] / 10.012444537618887 + 1)
                    y = int(point["location"]["y"] / 10.012444196428572 + 1)
                    z = int(point["location"]["z"] / 10.012444196428572 + 1)
                    assert 0 <= x < w
                    assert 0 <= y < h
                    assert 0 <= z < d

                    place_patch(output_mask[:, :, :, i], gaussian_patch, x, y, z)

        output_path = output_dir.joinpath(f"{exp_dir.stem}.npy")
        np.save(output_path, output_mask)


if __name__ == '__main__':
    main()
