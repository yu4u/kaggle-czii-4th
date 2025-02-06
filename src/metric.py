from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial import KDTree
from scipy.ndimage import maximum_filter


experiments = ["TS_5_4", "TS_69_2", "TS_6_4", "TS_6_6", "TS_73_6", "TS_86_3", "TS_99_9"]


particle_radius = {
    'apo-ferritin': 60,
    'beta-amylase': 65,
    'beta-galactosidase': 90,
    'ribosome': 150,
    'thyroglobulin': 130,
    'virus-like-particle': 135,
}


particle_to_weights = {
    'apo-ferritin': 1,
    'beta-amylase': 0,
    'beta-galactosidase': 2,
    'ribosome': 1,
    'thyroglobulin': 2,
    'virus-like-particle': 1,
}

particle_types = [
    "apo-ferritin",
    "beta-galactosidase",
    "ribosome",
    "thyroglobulin",
    "virus-like-particle",
    "beta-amylase",
]


def get_score(reference_points, reference_radius, candidate_points):
    beta = 4

    if len(reference_points) == 0:
        reference_points = np.array([])
        reference_radius = 1

    if len(candidate_points) == 0:
        candidate_points = np.array([])

    tp, fp, fn = compute_metrics(reference_points, reference_radius, candidate_points)
    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    fbeta = (1 + beta ** 2) * (precision * recall) / (beta ** 2 * precision + recall) if (
                                                                                                     precision + recall) > 0 else 0.0
    return fbeta, tp, fp, fn


class ParticipantVisibleError(Exception):
    pass


def compute_metrics(reference_points, reference_radius, candidate_points):
    num_reference_particles = len(reference_points)
    num_candidate_particles = len(candidate_points)

    if len(reference_points) == 0:
        return 0, num_candidate_particles, 0

    if len(candidate_points) == 0:
        return 0, 0, num_reference_particles

    ref_tree = KDTree(reference_points)
    candidate_tree = KDTree(candidate_points)
    raw_matches = candidate_tree.query_ball_tree(ref_tree, r=reference_radius)
    matches_within_threshold = []
    for match in raw_matches:
        matches_within_threshold.extend(match)
    # Prevent submitting multiple matches per particle.
    # This won't be be strictly correct in the (extremely rare) case where true particles
    # are very close to each other.
    matches_within_threshold = set(matches_within_threshold)
    tp = int(len(matches_within_threshold))
    fp = int(num_candidate_particles - tp)
    fn = int(num_reference_particles - tp)
    return tp, fp, fn


def score(
        solution: pd.DataFrame,
        submission: pd.DataFrame,
        distance_multiplier: float,
        beta: int,
        use_weight: bool = True) -> float:
    '''
    F_beta
      - a true positive occurs when
         - (a) the predicted location is within a threshold of the particle radius, and
         - (b) the correct `particle_type` is specified
      - raw results (TP, FP, FN) are aggregated across all experiments for each particle type
      - f_beta is calculated for each particle type
      - individual f_beta scores are weighted by particle type for final score
    '''
    _particle_radius = {k: v * distance_multiplier for k, v in particle_radius.items()}

    # Filter submission to only contain experiments found in the solution split
    split_experiments = set(solution['experiment'].unique())
    submission = submission.loc[submission['experiment'].isin(split_experiments)]

    # Only allow known particle types
    if not set(submission['particle_type'].unique()).issubset(set(particle_to_weights.keys())):
        raise ParticipantVisibleError('Unrecognized `particle_type`.')

    assert solution.duplicated(subset=['experiment', 'x', 'y', 'z']).sum() == 0
    assert _particle_radius.keys() == particle_to_weights.keys()

    results = {}
    for particle_type in solution['particle_type'].unique():
        results[particle_type] = {
            'total_tp': 0,
            'total_fp': 0,
            'total_fn': 0,
        }

    for experiment in split_experiments:
        for particle_type in solution['particle_type'].unique():
            reference_radius = _particle_radius[particle_type]
            select = (solution['experiment'] == experiment) & (solution['particle_type'] == particle_type)
            reference_points = solution.loc[select, ['x', 'y', 'z']].values

            select = (submission['experiment'] == experiment) & (submission['particle_type'] == particle_type)
            candidate_points = submission.loc[select, ['x', 'y', 'z']].values

            if len(reference_points) == 0:
                reference_points = np.array([])
                reference_radius = 1

            if len(candidate_points) == 0:
                candidate_points = np.array([])

            tp, fp, fn = compute_metrics(reference_points, reference_radius, candidate_points)

            results[particle_type]['total_tp'] += tp
            results[particle_type]['total_fp'] += fp
            results[particle_type]['total_fn'] += fn
    # print(results)
    aggregate_fbeta = 0.0
    for particle_type, totals in results.items():
        tp = totals['total_tp']
        fp = totals['total_fp']
        fn = totals['total_fn']

        precision = tp / (tp + fp) if tp + fp > 0 else 0
        recall = tp / (tp + fn) if tp + fn > 0 else 0
        fbeta = (1 + beta**2) * (precision * recall) / (beta**2 * precision + recall) if (precision + recall) > 0 else 0.0

        if use_weight:
            aggregate_fbeta += fbeta * particle_to_weights.get(particle_type, 1.0)
        else:
            aggregate_fbeta += fbeta

    if use_weight:
        aggregate_fbeta = aggregate_fbeta / sum(particle_to_weights.values())
    else:
        aggregate_fbeta = aggregate_fbeta / len(results)
    return aggregate_fbeta


def find_local_maxima(arrs, threshold, a):
    all_coordinates = []
    all_scores = []

    for i in range(arrs.shape[0]):
        # 最大フィルタを使用して近傍の最大値を取得
        arr = arrs[i]
        max_filtered = maximum_filter(arr, size=a, mode='constant')

        # 入力配列が最大フィルタの出力と等しい場所（ローカル最大値）を探す
        local_maxima = (arr == max_filtered) & (arr >= threshold)

        # ローカル最大値のインデックスを取得
        coordinates = np.argwhere(local_maxima)
        all_coordinates.append(coordinates)
        scores = arr[local_maxima]
        all_scores.append(scores)

    return all_coordinates, all_scores


def get_experiment_score(preds, fold_id):
    threshold = 0.1  # しきい値
    a = 3  # ピクセル数
    coordinates, scores = find_local_maxima(preds, threshold, a)
    best_ss = []

    csv_path = Path(__file__).parents[1].joinpath("output", "solution.csv")
    solution = pd.read_csv(csv_path, index_col=0)
    experiment = experiments[fold_id]
    solution = solution[solution["experiment"] == experiment]
    solution.head()

    for i, coordinate in enumerate(coordinates):
        particle_type = particle_types[i]
        score_i = scores[i]

        best_s = 0
        best_th = 0
        best_stat = None
        keep = (solution["particle_type"] == particle_type) & (solution["experiment"] == experiment)
        reference_points = solution[keep][["x", "y", "z"]].values
        reference_radius = particle_radius[particle_type] * 0.5

        for th in np.linspace(0.05, 0.95, 19):
            candidate_points = []

            for score_j, (z, y, x) in zip(score_i, coordinate):
                if score_j < th:
                    continue

                point = ((x + 0.5 - 1) * 10.012444537618887, (y + 0.5 - 1) * 10.012444196428572, (z + 0.5 - 1) * 10.012444196428572)
                candidate_points.append(point)

            s, tp, fp, fn = get_score(reference_points, reference_radius, candidate_points)
            stat = tp, fp, fn

            if s > best_s:
                best_s = s
                best_th = th
                best_stat = stat

        best_ss.append(best_s)

    return best_ss
