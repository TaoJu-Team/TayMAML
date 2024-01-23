import numpy as np
import playground.DAG.utils.feature_synthesize as fs


def features_extract_func(task):
    task_features = fs.task_features(task.job)
    return [task.task_config.cpu, task.task_config.memory,
            task.task_config.duration, task.waiting_task_instances_number,
            task_features[task.task_index]['first_layer_task'], task_features[task.task_index]['first_layer_instance'],
            task_features[task.task_index]['layers_task'], task_features[task.task_index]['child_task_numbers'],
            task_features[task.task_index]['child_instance_numbers']]


def features_extract_func_ac(task):
    return features_extract_func(task) + [task.task_config.instances_number, len(task.running_task_instances),
                                          len(task.finished_task_instances)]


def features_normalize_func(x):
    y = (np.array(x) - np.array([0, 0, 0, 0, 0, 0, 1.167, 1.167, 1.5, 1.833, 1.833])) / np.array(
        [2, 1, 1, 1, 100, 1, 0.897, 0.897, 0.957, 1.572, 1.572])
    return y


def features_normalize_func_ac(x):
    y = (np.array(x) - np.array([0, 0, 0, 0, 0, 0, 0, 0, 1.167, 1.167, 1.5, 1.833, 1.833, 0, 0, 0])) / \
        np.array([2, 1, 1, 1, 1, 1, 1, 1, 0.897, 0.897, 0.957, 1.572, 1.572, 1, 1, 1])
    # y = np.array(x)
    return y
