CLASS_LIST = [
    'bottle', 'cup', 'knife', 'bowl', 'wine glass', 'fork', 'spoon', 'banana',
    'apple', 'orange', 'cake', 'potted plant', 'mouse', 'keyboard', 'laptop',
    'cell phone', 'book', 'clock', 'chair', 'table', 'couch', 'bed', 'toilet',
    'tv', 'microwave', 'toaster', 'refrigerator', 'oven', 'sink', 'person',
    'background'
]

CLASS_IDS = {class_name: idx for idx, class_name in enumerate(CLASS_LIST)}

# Some helper synonyms, to handle cases where multiple words mean the same
# class. This list is used to sanitise class name lists both in ground truth &
# submitted maps
SYNONYMS = {
    'television': 'tv',
    'tvmonitor': 'tv',
    'tv monitor': 'tv',
    'computer monitor': 'tv',
    'coffee table': 'table',
    'dining table': 'table',
    'kitchen table': 'table',
    'desk': 'table',
    'stool': 'chair',
    'sofa': 'couch',
    'diningtable': 'dining table',
    'pottedplant': 'potted plant',
    'plant': 'potted plant',
    'cellphone': 'cell phone',
    'mobile phone': 'cell phone',
    'mobilephone': 'cell phone',
    'wineglass': 'wine glass',

    # background classes
    'none': 'background',
    'bg': 'background',
    '__background__': 'background'
}


def get_nearest_class_id(class_name):
    """
    Given a class string, find the id of that class
    This handles synonym lookup as well
    :param class_name: the name of the class being looked up (can be synonym from SYNONYMS)
    :return: an integer corresponding to nearest ID in CLASS_LIST, or None
    """
    class_name = class_name.lower()
    if class_name in CLASS_IDS:
        return CLASS_IDS[class_name]
    elif class_name in SYNONYMS:
        return CLASS_IDS[SYNONYMS[class_name]]
    return None


def get_nearest_class_name(class_name):
    """
    Given a string that might be a class name,
    return a string that is definitely a class name.
    Again, uses synonyms to map to known class names
    :param potential_class_name: the queried class name
    :return: the nearest class name from CLASS_LIST, or None
    """
    class_name = class_name.lower()
    if class_name in CLASS_IDS:
        return class_name
    elif class_name in SYNONYMS:
        return SYNONYMS[class_name]
    return None
