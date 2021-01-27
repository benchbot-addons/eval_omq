from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import gmean
import iou_tools

_IOU_TOOL = iou_tools.IoU()
_STATE_IDS = {"added": 0, "removed": 1, "constant": 2}


def create_scores(task_details,
                  environment_details,
                  scores_omq,
                  scores_avg_pairwise,
                  scores_avg_label,
                  scores_avg_spatial,
                  scores_avg_fp_quality,
                  scores_avg_state_quality=None):
    return {
        'task_details': task_details,
        'environment_details': environment_details,
        'scores': {
            'OMQ':
            scores_omq,
            'avg_pairwise':
            scores_avg_pairwise,
            'avg_label':
            scores_avg_label,
            'avg_spatial':
            scores_avg_spatial,
            'avg_fp_quality':
            scores_avg_fp_quality,
            **({} if scores_avg_state_quality is None else {
                   'avg_state_quality': scores_avg_state_quality
               })
        }
    }


def evaluate(results, ground_truths):
    return (evaluate_scd if results['task_details']['results_format']
            == 'object_map_with_states' else evaluate_semantic_slam)(
                results, ground_truths)


def evaluate_scd(results, ground_truths):
    o = OMQ()
    return create_scores(task_details=results['task_details'],
                         environment_details=results['environment_details'],
                         scores_omq=o.score([
                             (ground_truths[0]['ground_truth']['objects'],
                              results['results']['objects'])
                         ]),
                         scores_avg_pairwise=o.get_avg_overall_quality_score(),
                         scores_avg_label=o.get_avg_label_score(),
                         scores_avg_spatial=o.get_avg_spatial_score(),
                         scores_avg_fp_quality=o.get_avg_fp_score())


def evaluate_semantic_slam(results, ground_truths):
    pass


# NOTE For now we will ignore the concept of foreground and background quality in favor of
# spatial quality being just the IoU of a detection.


class OMQ(object):
    """
    Class for evaluating results from a Semantic SLAM system.
    Method is object map quality (OMQ)
    Based upon the probability-based detection quality (PDQ) system found below
    https://github.com/david2611/pdq_evaluation
    """
    def __init__(self, scd_mode=False):
        """
        Initialisation function for OMQ evaluator
        :param: scd_mode: flag for whether OMQ is evaluating a scene change detection system which has
        an extra consideration made for the uncertainty in the map for the state of an object (added, removed, same)
        """
        super(OMQ, self).__init__()
        self._tot_overall_quality = 0.0
        self._tot_spatial_quality = 0.0
        self._tot_label_quality = 0.0
        self._tot_fp_cost = 0.0
        self._tot_TP = 0
        self._tot_FP = 0
        self._tot_FN = 0
        self._tot_state_quality = 0.0
        self.scd_mode = scd_mode

    def reset(self):
        """
        Reset all internally stored evaluation measures to zero.
        :return: None
        """
        self._tot_overall_quality = 0.0
        self._tot_spatial_quality = 0.0
        self._tot_label_quality = 0.0
        self._tot_fp_cost = 0.0
        self._tot_TP = 0
        self._tot_FP = 0
        self._tot_FN = 0
        self._tot_state_quality = 0.0

    def add_map_eval(self, gt_objects, proposed_objects):
        """
        Adds a single map's object proposals and ground-truth to the overall evaluation analysis.
        :param gt_objects: list of ground-truth dictionaries present in the given map.
        :param proposed_objects: list of detection dictionaries objects provided for the given map
        :return: None
        """
        results = _calc_qual_map(gt_objects, proposed_objects, self.scd_mode)
        self._tot_overall_quality += results['overall']
        self._tot_spatial_quality += results['spatial']
        self._tot_label_quality += results['label']
        self._tot_fp_cost += results['fp_cost']
        self._tot_TP += results['TP']
        self._tot_FP += results['FP']
        self._tot_FN += results['FN']
        self._tot_state_quality += results['state_change']

    def get_current_score(self):
        """
        Get the current score for all scenes analysed at the current time.
        :return: The OMQ score across all maps as a float
        (average pairwise quality over the number of object-detection pairs observed with FPs weighted by confidence).
        """
        denominator = self._tot_TP + self._tot_fp_cost + self._tot_FN
        if denominator > 0.0:
            return self._tot_overall_quality / denominator
        return 0.0

    def score(self, param_lists):
        """
        Calculates the average quality score for a set of object proposals on
        a set of ground truth objects over a series of maps.
        The average is calculated as the average pairwise quality over the number of object-detection pairs observed
        with FPs weighted by confidence.
        Note that this removes any evaluation information that had been stored for previous maps.
        Assumes you want to score just the full list you are given.
        :param param_lists: A list of tuples where each tuple holds a list of ground-truth dicts and a list of
        detection dicts. Each map observed is an entry in the main list.
        :return: The OMQ score across all maps as a float
        """
        self.reset()

        for map_params in param_lists:
            map_results = self._get_map_evals(map_params)
            self._tot_overall_quality += map_results['overall']
            self._tot_spatial_quality += map_results['spatial']
            self._tot_label_quality += map_results['label']
            self._tot_fp_cost += map_results['fp_cost']
            self._tot_TP += map_results['TP']
            self._tot_FP += map_results['FP']
            self._tot_FN += map_results['FN']
            self._tot_state_quality += map_results['state_change']

        return self.get_current_score()

    def get_avg_spatial_score(self):
        """
        Get the average spatial quality score for all assigned object proposals in all maps analysed at the current time.
        Note that this is averaged over the number of assigned object proposals (TPs) and not the full set of TPs, FPs,
        and FNs like the final Semantic SLAM OMQ score.
        :return: average spatial quality of every assigned detection
        """
        if self._tot_TP > 0.0:
            return self._tot_spatial_quality / float(self._tot_TP)
        return 0.0

    def get_avg_label_score(self):
        """
        Get the average label quality score for all assigned object proposals in all maps analysed at the current time.
        Note that this is averaged over the number of assigned object proposals (TPs) and not the full set of TPs, FPs,
        and FNs like the final OMQ score.
        :return: average label quality of every assigned detection
        """
        if self._tot_TP > 0.0:
            return self._tot_label_quality / float(self._tot_TP)
        return 0.0

    def get_avg_state_score(self):
        """
        Get the average state quality score for all assigned object proposals in all maps analysed at the current time.
        Note that this is averaged over the number of assigned object proposals (TPs) and not the full set of TPs, FPs,
        and FNs like the final OMQ score.
        :return: average label quality of every assigned detection
        """
        if self._tot_TP > 0.0:
            return self._tot_state_quality / float(self._tot_TP)
        return 0.0

    def get_avg_overall_quality_score(self):
        """
        Get the average overall pairwise quality score for all assigned object proposals
        in all maps analysed at the current time.
        Note that this is averaged over the number of assigned object proposals (TPs) and not the full set of TPs, FPs,
        and FNs like the final OMQ score.
        :return: average overall pairwise quality of every assigned detection
        """
        if self._tot_TP > 0.0:
            return self._tot_overall_quality / float(self._tot_TP)
        return 0.0

    def get_avg_fp_score(self):
        """
        Get the average quality (1 - cost) for all false positive object proposals across all maps analysed at the
        current time.
        Note that this is averaged only over the number of FPs and not the full set of TPs, FPs, and FNs.
        Note that at present false positive cost/quality is based solely upon label scores.
        :return: average false positive quality score
        """
        if self._tot_FP > 0.0:
            return (self._tot_FP - self._tot_fp_cost) / self._tot_FP
        return 1.0

    def get_assignment_counts(self):
        """
        Get the total number of TPs, FPs, and FNs across all frames analysed at the current time.
        :return: tuple containing (TP, FP, FN)
        """
        return self._tot_TP, self._tot_FP, self._tot_FN

    def _get_map_evals(self, parameters):
        """
        Evaluate the results for a given image
        :param parameters: tuple containing list of ground-truth dicts and object proposal dicts
        :return: results dictionary containing total overall quality, total spatial quality on positively assigned
        object proposals, total label quality on positively assigned object proposals,
        total false positive cost for each false positive object proposal, number of true positives,
        number of false positives, number false negatives, and total state quality (relevant only in SCD).
        Format {'overall':<tot_overall_quality>, 'spatial': <tot_tp_spatial_quality>, 'label': <tot_tp_label_quality>,
        'fp_cost': <tot_false_positive_cost>, 'TP': <num_true_positives>, 'FP': <num_false_positives>,
        'FN': <num_false_positives>, 'state_change': <tot_state_quality>}
        """
        gt_objects, proposed_objects = parameters
        results = _calc_qual_map(gt_objects, proposed_objects, self.scd_mode)
        return results


def _vectorize_map_gts(gt_objects, scd_mode):
    """
    Vectorizes the required elements for all ground-truth object dicts as necessary for a given map.
    These elements are the ground-truth cuboids, class ids and state ids if given (0: added, 1: removed, 2: constant)
    :param gt_objects: list of all ground-truth dicts for a given image
    :return: (gt_cuboids, gt_labels, gt_state_ids).
    gt_cuboids: g length list of cuboid centroids and extents stored as dictionaries for each of the
    g ground-truth dicts. (dictionary format: {'centroid': <centroid>, 'extent': <extent>})
    gt_labels: g, numpy array of class labels as an integer for each of the g ground-truth dicts
    """
    gt_labels = np.array([gt_obj['class_id'] for gt_obj in gt_objects],
                         dtype=np.int)  # g,
    gt_cuboids = [{
        "centroid": gt_obj['centroid'],
        "extent": gt_obj["extent"]
    } for gt_obj in gt_objects]  # g,
    if scd_mode:
        gt_state_ids = [_STATE_IDS[gt_obj["state"]] for gt_obj in gt_objects]
    else:
        gt_state_ids = [-1 for gt_obj in gt_objects]
    return gt_cuboids, gt_labels, gt_state_ids


def _vectorize_map_props(proposed_objects, scd_mode):
    """
    Vectorize the required elements for all object proposal dicts as necessary for a given map.
    These elements are the detection boxes, the probability distributions for the labels, and the probability
    distribution of the state change estimations (if provided).
    :param proposed_objects: list of all proposed object dicts for a given image.
    :return: (prop_cuboids, prop_class_probs, det_)
    prop_cuboids: p length list of cuboid centroids and extents stored as dictionaries for the p object proposals
    (dictionary format: {'centroid': <centroid>, 'extent': <extent>})
    prop_class_probs: p x c numpy array of class label probability scores across all c classes for each of the
    p object proposals
    prop_state_probs: p x 3 numpy array of state probability scores across all 3 object states for each of the
    p object proposals. Note this is only relevant for SCD and states are (in order) [added, removed, same].
    """
    prop_class_probs = np.stack(
        [np.array(prop_obj['label_probs']) for prop_obj in proposed_objects],
        axis=0)  # d x c
    prop_cuboids = [{
        "centroid": prop_obj['centroid'],
        "extent": prop_obj["extent"]
    } for prop_obj in proposed_objects]  # d,
    if scd_mode:
        # Currently assuming that format of state_probs is 3 dimensional
        # format [<prob_added>, <prob_removed>, <prob_same>]
        prop_state_probs = np.stack([
            np.array(prop_obj['state_probs']) for prop_obj in proposed_objects
        ],
                                    axis=0)  # d x 3
    else:
        prop_state_probs = np.ones((len(proposed_objects), 3)) * -1
    return prop_cuboids, prop_class_probs, prop_state_probs


def _calc_spatial_qual(gt_cuboids, prop_cuboids):
    """
    Calculate the spatial quality for all object proposals on all ground truth objects for a given map.
    :param: gt_cuboids: g length list of all ground-truth cuboid dictionaries defining centroid and extent of objects
    :param: prop_cuboids: p length list of all proposed object cuboid dictionaries defining centroid and
    extent of objects
    :return: spatial_quality: g x p spatial quality score between zero and one for each possible combination of
    g ground truth objects and p object proposals.
    """
    # TODO optimize in some clever way in future rather than two for loops
    spatial_quality = np.zeros((len(gt_cuboids), len(prop_cuboids)),
                               dtype=np.float)  # g x d
    for gt_id, gt_cub_dict in enumerate(gt_cuboids):
        for prop_id, prop_cub_dict in enumerate(prop_cuboids):
            spatial_quality[gt_id, prop_id] = _IOU_TOOL.dict_iou(
                gt_cub_dict, prop_cub_dict)[1]

    return spatial_quality


def _calc_label_qual(gt_labels, prop_class_probs):
    """
    Calculate the label quality for all object proposals on all ground truth objects for a given image.
    :param gt_labels:  g, numpy array containing the class label as an integer for each ground-truth object.
    :param prop_class_probs: p x c numpy array of class label probability scores across all c classes
    for each of the p object proposals.
    :return: label_qual_mat: g x p label quality score between zero and one for each possible combination of
    g ground truth objects and p object proposals.
    """
    label_qual_mat = prop_class_probs[:,
                                      gt_labels].T.astype(np.float32)  # g x d
    return label_qual_mat


def _calc_state_change_qual(gt_state_ids, prop_state_probs):
    """
    Calculate the label quality for all object proposals on all ground truth objects for a given image.
    :param gt_state_ids:  g, numpy array containing the state label as an integer for each object.
    (0 = added, 1 = removed, 2 = same)
    :param prop_state_probs: p x 3 numpy array of label probability scores across all 3 states
    for each of the d object proposals.
    :return: state_qual_mat: g x p state quality score between zero and one for each possible combination of
    g ground truth objects and p object proposals.
    """
    # if we are not doing scd we return None
    if np.max(gt_state_ids) == -1:
        return None
    state_qual_mat = prop_state_probs[:, gt_state_ids].T.astype(
        np.float32)  # g x d
    return state_qual_mat


def _calc_overall_qual(label_qual_mat, spatial_qual_mat, state_qual_mat):
    """
    Calculate the overall quality for all object proposals on all ground truth objects for a given image
    :param label_qual_mat: g x p label quality score between zero and one for each possible combination of
    g ground truth objects and p object proposals.
    :param spatial_qual_mat: g x p spatial quality score between zero and one for each possible combination of
    g ground truth objects and p object proposals.
    :return: overall_qual_mat: g x p overall label quality between zero and one for each possible combination of
    g ground truth objects and p object proposals.
    """
    # For both the Semantic SLAM and SCD case, combined quality is the geometric mean of evaluated quality components
    if state_qual_mat is None:
        combined_mat = np.dstack((label_qual_mat, spatial_qual_mat))
    else:
        combined_mat = np.dstack(
            (label_qual_mat, spatial_qual_mat, state_qual_mat))

    # Calculate the geometric mean between all qualities
    # Note we ignore divide by zero warnings here for log(0) calculations internally.
    with np.errstate(divide='ignore'):
        overall_qual_mat = gmean(combined_mat, axis=2)

    return overall_qual_mat


def _gen_cost_tables(gt_objects, object_proposals, scd_mode):
    """
    Generate the cost tables containing the cost values (1 - quality) for each combination of ground truth objects and
    object proposals within a given map.
    :param gt_objects: list of all ground-truth object dicts for a given map.
    :param object_proposals: list of all object proposal dicts for a given map.
    :return: dictionary of g x p cost tables for each combination of ground truth objects and object proposals.
    Note that all costs are simply 1 - quality scores (required for Hungarian algorithm implementation)
    Format: {'overall': overall summary cost table, 'spatial': spatial quality cost table,
    'label': label quality cost table, 'state': state cost table}
    """
    # Initialise cost tables
    n_pairs = max(len(gt_objects), len(object_proposals))
    overall_cost_table = np.ones((n_pairs, n_pairs), dtype=np.float32)
    spatial_cost_table = np.ones((n_pairs, n_pairs), dtype=np.float32)
    label_cost_table = np.ones((n_pairs, n_pairs), dtype=np.float32)
    state_cost_table = np.ones((n_pairs, n_pairs), dtype=np.float32)

    # Generate all the matrices needed for calculations
    gt_cuboids, gt_labels, gt_state_ids = _vectorize_map_gts(
        gt_objects, scd_mode)
    prop_cuboids, prop_class_probs, prop_state_probs = _vectorize_map_props(
        object_proposals, scd_mode)

    # Calculate spatial, label and state qualities (state only used in SCD)
    label_qual_mat = _calc_label_qual(gt_labels, prop_class_probs)
    spatial_qual = _calc_spatial_qual(gt_cuboids, prop_cuboids)
    state_change_qual = _calc_state_change_qual(gt_state_ids, prop_state_probs)

    # Generate the overall cost table (1 - overall quality)
    overall_cost_table[:len(gt_objects), :len(object_proposals
                                              )] -= _calc_overall_qual(
                                                  label_qual_mat, spatial_qual,
                                                  state_change_qual)

    # Generate the spatial, label and (optionally) state cost tables
    spatial_cost_table[:len(gt_objects), :len(object_proposals
                                              )] -= spatial_qual
    label_cost_table[:len(gt_objects), :len(object_proposals
                                            )] -= label_qual_mat
    if scd_mode:
        state_cost_table[:len(gt_objects), :len(object_proposals
                                                )] -= state_change_qual

    return {
        'overall': overall_cost_table,
        'spatial': spatial_cost_table,
        'label': label_cost_table,
        'state': state_cost_table
    }


def _calc_qual_map(gt_objects, object_proposals, scd_mode):
    """
    Calculates the sum of qualities for the best matches between ground truth objects and object proposals for a map.
    Each ground truth object can only be matched to a single object proposal and vice versa as an gt-proposal pair.
    Note that if a ground truth object or proposal does not have a match, the quality is counted as zero.
    This represents a theoretical gt-proposal pair with the object or detection and a counterpart which
    does not describe it at all.
    Any provided proposal with a zero-quality match will be counted as a false positive (FP).
    Any ground-truth object with a zero-quality match will be counted as a false negative (FN).
    All other matches are counted as "true positives" (TP)
    If there are no ground-truth objects or object proposals for the map, the system returns zero and this map
    will not contribute to average score.
    :param gt_objects: list of ground-truth dictionaries describing the ground truth objects in the current map.
    :param object_proposals: list of object proposal dictionaries describing the object proposals for the current map.
    :return: results dictionary containing total overall spatial quality, total spatial quality on positively assigned
    object proposals, total label quality on positively assigned object proposals, total false positive cost,
    number of true positives, number of false positives, number false negatives, and total state change quality on
    positively assigned object proposals (relevant only for SCD).
    Format {'overall':<tot_overall_quality>, 'spatial': <tot_tp_spatial_quality>, 'label': <tot_tp_label_quality>,
    'fp_cost': <tot_fp_cost>, 'TP': <num_true_positives>, 'FP': <num_false_positives>, 'FN': <num_false_positives>,
    'state_change': <tot_tp_state_quality>}
    """

    tot_fp_cost = 0.0
    # if there are no object proposals or gt instances respectively the quality is zero
    if len(gt_objects) == 0 or len(object_proposals) == 0:
        if len(object_proposals) > 0:
            # Calculate FP quality
            # NOTE background class is the final class in the distribution which is ignored when calculating FP cost
            tot_fp_cost = np.sum([
                np.max(det_instance['label_probs'][:-1])
                for det_instance in object_proposals
            ])

        return {
            'overall': 0.0,
            'spatial': 0.0,
            'label': 0.0,
            'fp_cost': tot_fp_cost,
            'TP': 0,
            'FP': len(object_proposals),
            'FN': len(gt_objects),
            'state_change': 0.0
        }

    # For each possible pairing, calculate the quality of that pairing and convert it to a cost
    # to enable use of the Hungarian algorithm.
    cost_tables = _gen_cost_tables(gt_objects, object_proposals, scd_mode)

    # Use the Hungarian algorithm with the cost table to find the best match between ground truth
    # object and detection (lowest overall cost representing highest overall pairwise quality)
    row_idxs, col_idxs = linear_sum_assignment(cost_tables['overall'])

    # Transform the loss tables back into quality tables with values between 0 and 1
    overall_quality_table = 1 - cost_tables['overall']
    spatial_quality_table = 1 - cost_tables['spatial']
    label_quality_table = 1 - cost_tables['label']
    state_change_quality_table = 1 - cost_tables['state']

    # Go through all optimal assignments and summarize all pairwise statistics
    # Calculate the number of TPs, FPs, and FNs for the image during the process
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    false_positive_idxs = []

    for match_idx, match in enumerate(zip(row_idxs, col_idxs)):
        row_id, col_id = match
        # Handle "true positives"
        if overall_quality_table[row_id, col_id] > 0:
            true_positives += 1
        else:
            # Handle false negatives
            if row_id < len(gt_objects):
                false_negatives += 1
            # Handle false positives
            if col_id < len(object_proposals):
                # Check if false positive is actually a proposal of an isgroup object that has a better match
                # only ignored if best match otherwise would be isgroup object (non-zero quality) and max class matches
                if np.sum(overall_quality_table[:, col_id]) > 0:
                    # check if max match class is a grouped object
                    best_gt_idx = np.argmax(overall_quality_table[:, col_id])
                    if 'isgroup' in gt_objects[best_gt_idx] and gt_objects[
                            best_gt_idx]['isgroup']:
                        # check if the class of the proposal matches the class of the object
                        # (ignoring final class which should be background)
                        if np.argmax(
                                object_proposals[col_id]['label_probs']
                            [:-1]) == gt_objects[best_gt_idx]['class_id']:
                            # Check if at least 50% of the proposal is within the ground-truth object
                            if _IOU_TOOL.dict_prop_fraction(
                                    object_proposals[col_id],
                                    gt_objects[best_gt_idx]) >= 0.5:
                                # if all criteria met, skip this detection in both fp quality and number of fps
                                continue
                false_positives += 1
                false_positive_idxs.append(col_id)

    # Calculate the sum of quality at the best matching pairs to calculate total qualities for the image
    tot_overall_img_quality = np.sum(overall_quality_table[row_idxs, col_idxs])

    # Force spatial and label qualities to zero for total calculations as there is no actual association between
    # object proposals and therefore no TP when this is the case.
    spatial_quality_table[overall_quality_table == 0] = 0.0
    label_quality_table[overall_quality_table == 0] = 0.0
    state_change_quality_table[overall_quality_table == 0] = 0.0

    # Calculate the sum of spatial and label qualities only for TP samples
    tot_tp_spatial_quality = np.sum(spatial_quality_table[row_idxs, col_idxs])
    tot_tp_label_quality = np.sum(label_quality_table[row_idxs, col_idxs])
    tot_tp_state_change_quality = np.sum(state_change_quality_table[row_idxs,
                                                                    col_idxs])

    # Calculate the penalty for assigning a high label probability to false positives
    # NOTE background class is final class in the class list and is not considered
    # This will be the geometric mean between the maximum label quality and maximum state estimated (ignore same)
    if scd_mode:
        with np.errstate(divide='ignore'):
            tot_fp_cost = np.sum([
                gmean([
                    np.max(object_proposals[i]['label_probs'][:-1]),
                    np.max(object_proposals[i]['state_probs'][:-1])
                ]) for i in false_positive_idxs
            ])
    else:
        tot_fp_cost = np.sum([
            np.max(object_proposals[i]['label_probs'][:-1])
            for i in false_positive_idxs
        ])

    return {
        'overall': tot_overall_img_quality,
        'spatial': tot_tp_spatial_quality,
        'label': tot_tp_label_quality,
        'fp_cost': tot_fp_cost,
        'TP': true_positives,
        'FP': false_positives,
        'FN': false_negatives,
        'state_change': tot_tp_state_change_quality
    }
