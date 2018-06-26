import numpy as np
import os
import random

class Dataset(object):

    def __init__(self, data_source, which_set='train', task_type='classification'):

        self.data_source = data_source
        self.task_type = task_type # classification | regression | density estimation
        self.which_set = which_set

    def _sample_mini_dataset(self, num_shots, num_classes=1):
        if self.task_type == 'regression':
            assert num_classes==1, "num_classes={0} in regression task".format(num_classes)
        classes = self.data_source.sample_classes(num_classes=num_classes, which_set=self.which_set)
        for class_idx, class_obj in enumerate(classes):
            for sample in class_obj.sample(num_shots):
                if self.task_type == 'regression':
                    yield (sample[:-1], sample[-1])
                else:
                    yield (sample, class_idx)

    def _split_train_test(self, samples, test_shots):
        train_set = list(samples)
        if self.task_type == 'regression':
            test_set = train_set[:test_shots]
            train_set = train_set[test_shots:]
            return train_set, test_set
        test_set = []
        labels = set(item[1] for item in train_set)
        for _ in range(test_shots):
            for label in labels:
                for i, item in enumerate(train_set):
                    if item[1] == label:
                        del train_set[i]
                        test_set.append(item)
                        break
        if len(test_set) < len(labels) * test_shots:
            raise IndexError('not enough examples of each class for test set')
        return train_set, test_set


    def sample_task(self, num_shots, num_classes, use_split=False, test_shots=1):
        mini_dataset = self._sample_mini_dataset(num_shots, num_classes)
        if use_split:
            return self._split_train_test(mini_dataset, test_shots=test_shots)
        else:
            return mini_dataset


    # def sample_mini_dataset(self, num_shots, num_classes=1):
    #     if self.task_type == 'classification':
    #         for class_idx, class_obj in enumerate(self.data_source.sample_classes(num_classes)):
    #             for sample in class_obj.sample(num_shots):
    #                 yield (sample, class_idx)
    #     elif self.task_type == 'regression':
    #         assert num_classes==1, "num_classes can only be 1 in regression task"
    #         class_obj = next(self.data_source.sample_classes(1))
    #         for sample in class_obj.sample(num_shots):
    #             yield (sample[0], sample[1])
    #     elif self.task_type == 'density estimation':
    #         pass
    #
    # def split_train_test(self, samples, test_shots=1):
    #     train_set = list(samples)
    #     if self.task_type == 'regression':
    #         test_set = train_set[:test_shots]
    #         train_set = train_set[test_shots:]
    #         return train_set, test_set
    #     test_set = []
    #     labels = set(item[1] for item in train_set)
    #     for _ in range(test_shots):
    #         for label in labels:
    #             for i, item in enumerate(train_set):
    #                 if item[1] == label:
    #                     del train_set[i]
    #                     test_set.append(item)
    #                     break
    #     if len(test_set) < len(labels) * test_shots:
    #         raise IndexError('not enough examples of each class for test set')
    #     return train_set, test_set

    def mini_batch(self, samples, batch_size, num_batches, replacement):
        samples = list(samples)
        if replacement:
            for _ in range(num_batches):
                yield random.sample(samples, batch_size)
            return
        cur_batch = []
        batch_count = 0
        while True:
            random.shuffle(samples)
            for sample in samples:
                cur_batch.append(sample)
                if len(cur_batch) < batch_size:
                    continue
                yield cur_batch
                cur_batch = []
                batch_count += 1
                if batch_count == num_batches:
                    return
