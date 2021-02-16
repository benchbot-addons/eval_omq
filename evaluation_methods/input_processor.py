import warnings
import numpy as np

_BACKGROUND_SYNONYMS = {
    'none': 'background',
    'bg': 'background',
    '__background__': 'background'}

_TYPE_SEMANTIC_SLAM = 'semantic_slam'
_TYPE_SCD = 'scd'

class InputProcessor(object):
    def __init__(self, class_list, synonyms=_BACKGROUND_SYNONYMS):
        self.class_list = class_list 
        self.synonyms = synonyms

        # TODO need a better way to guarantee that background will be part of the class list
        # There should always be a background class which is the final label in the class list
        background_idx = [id for id, val in enumerate(class_list) 
                          if val == self.get_nearest_class_name("background")]
        # Throw an error if too many classes are labelled as background
        if len(background_idx) > 1:
            raise ValueError(f"class_list has {len(background_idx)} background classes (should be 1)")
        # If there is no background class in the class list then append one to the end
        if len(background_idx) == 0:
            self.class_list += ["background"]
        # Otherwise move the background class to the end of the list if it isn't there already
        elif background_idx[0] != len(class_list)-1:
            self.class_list += [self.class_list.pop(background_idx[0])]
        
        self.class_ids = {class_name: idx for idx, class_name in enumerate(self.class_list)}
    
    def process_gt(self, gt_data):
        # go through all gt sets we need to evaluate
        for gt_dict in gt_data:
            # go through all objects in the gt object map
            for gt_obj in gt_dict['ground_truth']['objects']:
                # add class_id parameter to the object parameters in the dict
                gt_obj['class_id'] = self.get_nearest_class_id(gt_obj['class'])
        return gt_data

    def process_results(self, results_data):
      # NOTE this might not be the best thing to have
      is_scd = _TYPE_SCD in results_data['task_details']['name']

      # TODO I think this has been moved elsewhere. Should check.
      # Validate the provided results data
      # Evaluator._validate_results_data(results_data)
      # for i, o in enumerate(results_data['objects']):
      #     Evaluator._validate_object_data(o, i, scd=is_scd)

      # Use the default class_list if none is provided
      if 'class_list' not in results_data['results'] or not results_data['results']['class_list']:
        warnings.warn(
            "No 'class_list' field provided; assuming results have used "
            "ground truth class list")
        results_data[results]['class_list'] = self.class_list
      
      # Sanitise all probability distributions for labels & states if
      # applicable (sanitising involves dumping unused bins to the background
      # / uncertain class, normalising the total probability to 1, &
      # optionally rearranging to match a required order)
      for i, o in enumerate(results_data['results']['objects']):
        if len(o['label_probs']) != len(results_data['results']['class_list']):
          raise ValueError(
              "The label probability distribution for object %d has a "
              "different length (%d) \nto the used class list (%d). " %
              (i, len(o['label_probs']), len(
                  results_data['results']['class_list'])))
        o['label_probs'] = self._sanitise_prob_dist(
          o['label_probs'], results_data['results']['class_list'])
        if is_scd:
          o['state_probs'] = self._sanitise_prob_dist(
              o['state_probs'])

      # We have applied the ground_truth class list to the label probs, so update
      # the class list in results_data
      results_data['results']['class_list'] = self.class_list

      return results_data


    def get_nearest_class_id(self, class_name):
        """
        Given a class string, find the id of that class
        This handles synonym lookup as well
        :param class_name: the name of the class being looked up (can be synonym from SYNONYMS)
        :return: an integer corresponding to nearest ID in CLASS_LIST, or None
        """
        class_name = class_name.lower()
        if class_name in self.class_ids:
            return self.class_ids[class_name]
        elif class_name in self.synonyms:
            return self.class_ids[self.synonyms[class_name]]
        return None


    def get_nearest_class_name(self, class_name):
        """
        Given a string that might be a class name,
        return a string that is definitely a class name.
        Again, uses synonyms to map to known class names
        :param potential_class_name: the queried class name
        :return: the nearest class name from CLASS_LIST, or None
        """
        class_name = class_name.lower()
        if class_name in self.class_list:
            return class_name
        elif class_name in self.synonyms:
            return self.synonyms[class_name]
        return None

    def _sanitise_prob_dist(self, prob_dist, current_class_list=None):
            # This code makes the assumption that the last bin is the background /
            # "I'm not sure" class (it is an assumption because this function can
            # be called with no explicit use of a class list)
            BACKGROUND_CLASS_INDEX = -1

            # Create a new prob_dist if we were given a current class list by
            # converting all current classes to items in our current class
            # list, & amalgamating all duplicate values (e.g. anything not
            # found in our list will be added to the background class)
            if current_class_list is not None:
                new_prob_dist = [0.0] * len(self.class_list)
                for i, c in enumerate(current_class_list):
                    new_prob_dist[BACKGROUND_CLASS_INDEX if self.
                                  get_nearest_class_id(c) is None else self.
                                  get_nearest_class_id(c)] += prob_dist[i]
                prob_dist = new_prob_dist

            # Either normalize the distribution if it has a total > 1, or dump
            # missing probability into the background / "I'm not sure" class
            total_prob = np.sum(prob_dist)
            if total_prob > 1:
                prob_dist /= total_prob
            else:
                prob_dist[BACKGROUND_CLASS_INDEX] += 1 - total_prob

            return prob_dist
