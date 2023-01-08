"""
Credits to Kenza Amara (kamara@student.ethz.ch) and the authors of OneForest
"""
from typing import Callable, List

import numpy as np
import ot
from scipy.optimize import linear_sum_assignment

from .datatypes import DeepforestDetection, FieldData, MatchedFieldData


def calculate_ot_map(
    ot_algo: Callable,
    coord_pred: np.ndarray, coord_ground_truth: np.ndarray,
    proba_pred: np.ndarray, proba_ground_truth: np.ndarray,
    mu: float = 0.5, regularizer: float = 1e-2,
) -> np.ndarray:
    """
    Calculate OT map
    
    Parameters
    ----------
    ot_algo: Callable
        Optimal transport algorithm to use. E.g. ot.bregman.sinkhorn
    coord_pred: np.ndarray
        Coordinates of detected trees
    coord_ground_truth: np.ndarray
        Coordinates of field data measurements
    proba_pred: np.ndarray
        Predicted class (tree group) of detected trees
    proba_ground_truth: np.ndarray
        Ground truth of tree groups (from field data measurements)
    mu: float
        If 0, only the relative distance is used to calculate the OT map
        If 1, only the cross entropy of the tre group is used to calculate the OT map
    """
    assert coord_pred.shape[1] == 2
    assert coord_ground_truth.shape[1] == 2
    assert proba_pred.shape[1] == 1
    assert proba_ground_truth.shape[1] == 1
    num_pred, num_ground_truth = coord_pred.shape[0], coord_ground_truth.shape[0]

    pairwise_dist = ot.dist(coord_pred, coord_ground_truth)
    pairwise_ce = _calculate_pairwise_binary_crossentropy(proba_pred, proba_ground_truth)

    weight_mat_dist = _standaradize_matrix(pairwise_dist)
    weight_mat_ce = _standaradize_matrix(pairwise_ce)
    weight_mat = (1 - mu) * weight_mat_dist + mu * weight_mat_ce

    hist_pred = np.ones((num_pred, )) / num_pred
    hist_ground_truth = np.ones((num_ground_truth, )) / num_ground_truth

    ot_plan = ot_algo(hist_pred, hist_ground_truth, weight_mat, regularizer)
    return ot_plan


def match_detections_to_field_data(
    tree_detections: List[DeepforestDetection], field_data: List[FieldData],
    ot_plan: np.ndarray, greedy: bool = True
) -> List[MatchedFieldData]:
    """ Given OT map, matches detections to field data"""
    if greedy:
        cost_matrix = - ot_plan * 1e5
        row_idx, col_idx = linear_sum_assignment(cost_matrix)
        tree_detections = [tree_detections[i] for i in row_idx]
        matching_idx = col_idx
    else:
        matching_idx = np.argmax(ot_plan, axis=0)
    print(f"Number of detection: {len(tree_detections)}")
    print(f"Number of field data measurements: {len(field_data)}")
    print(f"Number of unique matches: {len(np.unique(matching_idx))}")
    matches = []
    for i, matched_field_data in zip(matching_idx, field_data):
        matched_detection = tree_detections[i]
        match = MatchedFieldData(
            matched_field_data.name,
            matched_field_data.group,
            matched_field_data.lat,
            matched_field_data.lon,
            matched_field_data.diameter,
            matched_field_data.updated_diameter,
            matched_field_data.height,
            matched_field_data.year,
            matched_field_data.plot_id,
            matched_field_data.site,
            matched_field_data.x,
            matched_field_data.y,
            matched_field_data.AGB,
            matched_field_data.carbon,
            matched_detection
        )
        matches.append(match)
    return matches


def _calculate_pairwise_binary_crossentropy(proba_pred: np.ndarray, proba_ground_truth: np.ndarray) -> np.ndarray:
    num_pred = len(proba_pred)
    num_ground_truth = len(proba_ground_truth)
    pairwise_ce = np.zeros((num_pred, num_ground_truth))
    for i in range(num_pred):
        for j in range(num_ground_truth):
            if proba_ground_truth[j] == 1:
                if proba_pred[i] < 0.0001:
                    proba_pred[i] = 0.0001
                pairwise_ce[i][j] = - np.log(proba_pred[i])
            elif proba_ground_truth[j] == 0:
                if proba_pred[i] > 0.9999:
                    proba_pred[i] = 0.9999
                pairwise_ce[i][j] = - np.log(1 - proba_pred[i])
            else:
                raise ValueError("proba_ground_truth should contain binary values")
    return pairwise_ce


def _standaradize_matrix(mat: np.ndarray) -> np.ndarray:
    min_value, max_value = mat.min(), mat.max()
    return (mat - min_value) / (max_value - min_value)
