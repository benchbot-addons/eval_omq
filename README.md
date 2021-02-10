# BenchBot Add-on: Object Map Quality (OMQ) evaluation method

This add-on contains an implementation of the Object Map Quality (OMQ) method for numerically evaluating the quality of an object map with respect to another ground truth object map.

The implementation is in Python, and is used through the `benchbot_eval` script.

## The results format

Results for both semantic SLAM and scene change detection tasks consist of an object-based semantic map, and associated task metadata. Results from the two types of task differ only in that objects in scene change detection tasks require a probability distribution describing the suggested state change (`'state_probs'`). See further below for more details.

Our results format defines an object-based semantic map as a list of objects with information about both their semantic label and spatial geometry:

- Semantic label information is provided through a probability distribution (`'label_probs'`) across either a provided (`'class_list'`), or our default class list:
  ```
  bottle, cup, knife, bowl, wine glass, fork, spoon, banana, apple, orange, cake,
  potted plant, mouse, keyboard, laptop, cell phone, book, clock, chair, table,
  couch, bed, toilet, tv, microwave, toaster, refrigerator, oven, sink, person,
  background
  ```
- Spatial geometry information is provided by describing the location (`'centroid'`) of a cuboid in 3D space, whose dimensions are `'extent'`.

The entire required results format is outlined below:

```
{
    'task_details': {
        'type': <test_challenge_type>,
        'control_mode': <test_control_mode>,
        'localisation_mode': <test_localisation_mode>
    },
    'environment_details': {
        'name': <test_environment_name>,
        'numbers': [<test_environment_numbers>]
    },
    'objects': [
        {
            'label_probs': [<object_class_probability_distribution>],
            'centroid': [<xc>, <yc>, <zc>],
            'extent': [<xe>, <ye>, <ze>],
            'state_probs': [<pa>, <pr>, <pu>]
        },
        ...
    ]
    'class_list': [<classes_order>]
}
```

Notes:

- `'task_details'` and `'environment_details'` define what task was being solved when the results file was produced, and what environment the agent was operating in. These values are defined in the BenchBot software stack when `benchbot_run` is executed to start a new task. For example:

  ```
  u@pc:~$ benchbot_run --task semantic_slam:passive:ground_truth --env miniroom:1
  ```

  would produce the following:

  ```json
  "task_details": {
    "type": "semantic_slam",
    "control_mode": "passive",
    "localisation_mode": "ground_truth"
  },
  "environment_details": {
    "name": "miniroom",
    "numbers": [1]
  }
  ```

  The above dicts can be obtained at runtime through the `BenchBot.task_details` and `BenchBot.environment_details` [API properties](https://github.com/roboticvisionorg/benchbot_api).

- For `'task_details'`:
  - `'type'` must be either `'semantic_slam'` or `'scd'`
  - `'control_mode'` must be either `'passive'` or `'active'`
  - `'localisation_mode'` must be either `'ground_truth'` or `'dead_reckonoing'`
- For `'environment_details'`:
  - `'name'` must be present (any name is accepted, but will fail at evaluation time if an associated ground truth cannot be found)
  - `'numbers'` must be a list of integers (or strings which can be converted to integers) between 1 and 5 inclusive
- For each object in `'objects'` the objects list:
  - `'label_probs'` is the probability distribution for the suggested object label corresponding to the class list in `'class_list'`, or our default class list above (must be a list of numbers)
  - `'centroid'` is the 3D coordinates for the centre of the object's cuboid (must be a list of 3 numbers)
  - `'extent'` is the **full** width, height, and depth of the cuboid (must be a list of 3 numbers)
    - **Note** the cuboid described by `'centroid'` and `'extent'` must be axis-aligned in global coordinates, and use metres for units
  - `'state_probs'` must be a list of 3 numbers corresponding to the probability that the object was added, removed, or changed respectively (**only** required when `'type'` is `'scd'` in `'task_details'`)
    - **Note** if your system is not probabilistic, simply use all 0s and a single 1 for any of the distributions above (e.g. `'state_probs'` of `[1, 0, 0]` for an added object)
- `'class_list'` is a list of strings defining a custom order for the probabilities in the `'label_probs'` distribution field of objects (if not provided the default class list and order is assumed). Other notes on `class_list`:
  - there is some support given for class name synonyms in `./benchbot_eval/class_list.py`.
  - any class names given that are not in our class list, and don't have an appropriate synonym, will have their probability added to the `'background'` class (this avoids over-weighting label predictions solely because your detector had classes we don't support)
- all probability distributions in `'label_probs'` and `'state_probs'` are normalized if their total probability is greater than 1, or have the missing probability added to the final class (`'background'` or `'unchanged'`)

## Generating results for evaluation

An algorithm attempting to solve a semantic scene understanding task only has to fill in the list of `'objects'` and the `'class_list'` field (only if a custom class list has been used); everything else can be pre-populated using the [provided BenchBot API methods](https://github.com/roboticvisionorg/benchbot_api). Using these helper methods, only a few lines of code is needed to create results that can be used with our evaluator:

```python
from benchbot_api import BenchBot

b = BenchBot()

my_map = ...  # TODO solve the task, & get a map with a set of objects

results = b.empty_results()  # new results with pre-populated metadata
for o in my_map.objects:
    new_obj = b.empty_object()  # new object result with correct fields
    new_obj[...] = ...  # TODO insert data from object in my_map
    results['objects'].append(new_obj)  # add new object result to results
```

Alternatively, users of the `Agent` class can use the data provided in the `Agent.save_result()` function call:

```python
from benchbot_api import Agent

class MyAgent(Agent):
    def __init__(self):
        ...
        my_map = ...

    ...

    def save_result(self, filename, empty_results, empty_object_fn):
        for o in self.my_map:
            new_obj = empty_object_fn()  # new object result with correct fields
            new_obj[...] = ... # TODO insert data from object in my_map
            empty_results['objects'].append(new_obj)  # add new object result to results

        # TODO write the results to 'filename'...

```

## Specific details of the evaluation process

Solutions to our semantic scene understanding tasks require the generation of an object-based semantic map for given environments that the robot has explored. The tasks merely differ in whether the object-based semantic map is of all the objects in an environment, or the objects that have changed between two scenes of an environment.

### Object Map Quality (OMQ)

To evaluate object-based semantic maps we introduce a novel metric called Object Map Quality (OMQ). Object map quality compares the object cuboids of a ground-truth object-based semantic map with the object cuboids provided by a generated object-based semantic map. The metric compares both geometric overlap of the generated map with the ground-truth map, and the accuracy of the semantically labelling in the generated map.

The steps for calculating OMQ for a generated object-based semantic map are as follows:

1. Compare each object in the generated map to all ground-truth objects, by calculating a _pairwise object quality_ score for each pair. The pairwise object quality is the geometric mean of all object sub-qualities. Standard OMQ has two object sub-quality scores:

   - _spatial quality_ is the 3D IoU of the ground-truth and generated cuboids being compared
   - _label quality_ is the probability assigned to the class label of the ground-truth object being compared to

   Note that the pairwise object quality will be zero if either sub-quality score is zero - like when there is no overlap between object cuboids.

2. Each object in the generated map is assigned the ground-truth object with the highest non-zero pairwise quality, establishing the _"true positives"_ (some non-zero quality), _false negatives_ (ground-truth objects with no pairwise quality match), and _false positives_ (objects in the generated map with no pairwise quality match).

3. A _false positive cost_, defined as the maximum confidence given to a non-background class, is given for all false positive objects in the generated map

4. Overall OMQ score is calculated as the sum of all "true positive" qualities divided by the sum of: number of "true positives", number of false negatives, and total false positive cost

Notes:

- Average pairwise qualities for the set of "true positive" objects are often also provided with the overall OMQ score. Average pairwise qualities include an average overall quality, as well as averages for each of the object sub-qualities (spatial and label for standard OMQ)
- OMQ is based on the probabilistic object detection quality measure PDQ, which is described in [our paper](http://openaccess.thecvf.com/content_WACV_2020/papers/Hall_Probabilistic_Object_Detection_Definition_and_Evaluation_WACV_2020_paper.pdf) and [accompanying code](https://github.com/david2611/pdq_evaluation)).

### Evaluating Semantic SLAM with OMQ

![semantic_slam_object_map](./.docs/semantic_slam_obmap.png)

Evaluation compares the object-based semantic map (as shown above) generated by the robot with the ground-truth map of the environment using OMQ. The evaluation metric is used exactly as described above.

### Evaluating scene change detection (SCD) with OMQ

![scene_change_detection_object_map](./.docs/scd_obmap.png)

Scene change detection (SCD) creates object-based semantic maps comprising of the objects that have _changed_ between two scenes. Valid changes are the addition or removal of an object, with unchanged provided as a third state to capture uncertainty in state change.

Evaluation of SCD compares the object-based semantic map of changed objects generated by the robot with the ground-truth map of object changes between two scenes in an environment. The comparison is done using OMQ, but a third pairwise object sub-quality is added to capture the quality of the detected state change:

- _state quality_ is the probability given to the correct change state (e.g. [0.4, 0.5, 0.1] on an added object would get a state score of 0.4)
- _final pairwise quality_ ([step 1 of OMQ](<#object-map-quality-(omq)>)) is now the geometric mean of three sub-quality scores (spatial, label, and state)
- _false positive cost_ ([step 3 of OMQ](<#object-map-quality-(omq)>)) is now the geometric mean of both the maximum label confidence given to a non-background class, and the maximum state confidence of a added or removed state change (i.e. not unchanged). This means both overconfidence in label and state change will increase the false positive cost.
- **Note:** including state quality changes the quality scores for pairs of generated and ground-truth objects due to averaging over 3 terms instead of 2 (i.e. pairwise scores will be different between semantic SLAM and SCD tasks)
