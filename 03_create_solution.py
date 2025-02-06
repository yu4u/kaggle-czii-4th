from pathlib import Path
import json
import pandas as pd

from src.metric import particle_types


def main():
    root_dir = Path(__file__).parent.joinpath("input")
    output_dir = Path(__file__).parent.joinpath("output")
    output_dir.mkdir(exist_ok=True, parents=True)

    json_names = [particle_type + ".json" for particle_type in particle_types]
    rows = []

    for exp_dir in root_dir.joinpath("train", "overlay", "ExperimentRuns").iterdir():
        if not exp_dir.is_dir():
            continue

        json_dir = exp_dir.joinpath("Picks")
        experiment = exp_dir.stem

        for i, json_name in enumerate(json_names):
            json_path = json_dir.joinpath(json_name)
            particle_type = json_name.split(".")[0]

            with open(json_path) as f:
                picks = json.load(f)

                for point in picks["points"]:
                    x = point["location"]["x"]
                    y = point["location"]["y"]
                    z = point["location"]["z"]
                    rows.append([experiment, particle_type, x, y, z])

    df = pd.DataFrame(rows, columns=["experiment", "particle_type", "x", "y", "z"])
    df.index.name = "id"
    df.to_csv(output_dir.joinpath("solution.csv"))


if __name__ == '__main__':
    main()
