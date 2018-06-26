import random
import tensorflow as tf
from misc.variables import (interpolate_vars, average_vars, subtract_vars, add_vars, scale_vars,
                        VariableState)
import components.metrics as metrics
import numpy as np


class MetaLearner(object):

    def __init__(self, session, variables=None, transductive=False, pre_step_op=None):
        self.session = session
        task_type = 'classification'
        assert task_type in ['classification', 'regression', 'density estimation'], "task_type {0} not supported".format(task_type)
        self.task_type = task_type
        self._model_state = VariableState(self.session, variables or tf.trainable_variables())
        self._full_state = VariableState(self.session,
                                         tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))
        self._transductive = transductive
        self._pre_step_op = pre_step_op

        if self.task_type == 'classification':
            self.eval_metric = metrics.num_correct
        elif self.task_type == 'regression':
            self.eval_metric == metrics.mse
        elif self.task_type == 'density estimation':
            self.eval_metric = None

    def evaluate(self,
                 dataset,
                 input_ph,
                 label_ph,
                 minimize_op,
                 predictions,
                 num_classes,
                 num_shots,
                 inner_batch_size,
                 inner_iters,
                 replacement):
        # train_set, test_set = _split_train_test(
        #     _sample_mini_dataset(dataset, num_classes, num_shots+1))
        train_set, test_set = dataset.sample_task(num_shots+1, num_classes, use_split=True, test_shots=1)
        old_vars = self._full_state.export_variables()
        mini_batches = dataset.mini_batch(train_set, inner_batch_size, inner_iters, replacement)
        for batch in mini_batches:
            inputs, labels = zip(*batch)
            if self._pre_step_op:
                self.session.run(self._pre_step_op)
            self.session.run(minimize_op, feed_dict={input_ph: inputs, label_ph: labels})
        test_preds = self._test_predictions(train_set, test_set, input_ph, predictions)
        #num_correct = 0 #sum([pred == sample[1] for pred, sample in zip(test_preds, test_set)])
        e = self.eval_metric(test_preds, labels)
        self._full_state.import_variables(old_vars)
        return e

    def _test_predictions(self, train_set, test_set, input_ph, predictions):
        if self._transductive:
            inputs, _ = zip(*test_set)
            return self.session.run(predictions, feed_dict={input_ph: inputs})
        res = []
        for test_sample in test_set:
            inputs, _ = zip(*train_set)
            inputs += (test_sample[0],)
            res.append(self.session.run(predictions, feed_dict={input_ph: inputs})[-1])
        return res

class Reptile(MetaLearner):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def train_step(self,
                   dataset,
                   input_ph,
                   label_ph,
                   minimize_op,
                   num_classes,
                   num_shots,
                   inner_batch_size,
                   inner_iters,
                   replacement,
                   meta_step_size,
                   meta_batch_size):
        """
        Perform a Reptile training step.

        Args:
          dataset: a sequence of data classes, where each data
            class has a sample(n) method.
          input_ph: placeholder for a batch of samples.
          label_ph: placeholder for a batch of labels.
          minimize_op: TensorFlow Op to minimize a loss on the
            batch specified by input_ph and label_ph.
          num_classes: number of data classes to sample.
          num_shots: number of examples per data class.
          inner_batch_size: batch size for every inner-loop
            training iteration.
          inner_iters: number of inner-loop iterations.
          replacement: sample with replacement.
          meta_step_size: interpolation coefficient.
          meta_batch_size: how many inner-loops to run.
        """
        old_vars = self._model_state.export_variables()
        new_vars = []
        for _ in range(meta_batch_size):
            mini_dataset = dataset.sample_task(num_shots, num_classes)
            mini_batches = dataset.mini_batch(mini_dataset, inner_batch_size, inner_iters,
                                              replacement)
            for batch in mini_batches:
                inputs, labels = zip(*batch)
                if self._pre_step_op:
                    self.session.run(self._pre_step_op)
                self.session.run(minimize_op, feed_dict={input_ph: inputs, label_ph: labels})
            new_vars.append(self._model_state.export_variables())
            self._model_state.import_variables(old_vars)
        new_vars = average_vars(new_vars)
        self._model_state.import_variables(interpolate_vars(old_vars, new_vars, meta_step_size))

class FOML(MetaLearner):

    def __init__(self, *args, tail_shots=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.tail_shots = tail_shots

    def train_step(self,
                   dataset,
                   input_ph,
                   label_ph,
                   minimize_op,
                   num_classes,
                   num_shots,
                   inner_batch_size,
                   inner_iters,
                   replacement,
                   meta_step_size,
                   meta_batch_size):
        old_vars = self._model_state.export_variables()
        updates = []
        for _ in range(meta_batch_size):
            mini_dataset = dataset.sample_task(num_shots, num_classes)
            mini_batches = dataset.mini_batch(mini_dataset, inner_batch_size, inner_iters,
                                              replacement)
            for batch in mini_batches:
                inputs, labels = zip(*batch)
                last_backup = self._model_state.export_variables()
                if self._pre_step_op:
                    self.session.run(self._pre_step_op)
                self.session.run(minimize_op, feed_dict={input_ph: inputs, label_ph: labels})
            updates.append(subtract_vars(self._model_state.export_variables(), last_backup))
            self._model_state.import_variables(old_vars)
        update = average_vars(updates)
        self._model_state.import_variables(add_vars(old_vars, scale_vars(update, meta_step_size)))

    def _mini_batches(self, mini_dataset, inner_batch_size, inner_iters, replacement):
        """
        Generate inner-loop mini-batches for the task.
        """
        if self.tail_shots is None:
            for value in _mini_batches(mini_dataset, inner_batch_size, inner_iters, replacement):
                yield value
            return
        train, tail = _split_train_test(mini_dataset, test_shots=self.tail_shots)
        for batch in _mini_batches(train, inner_batch_size, inner_iters - 1, replacement):
            yield batch
        yield tail

class MAML(MetaLearner):

    def __init__(self, *args, tail_shots=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.tail_shots = tail_shots

    def train_step(self,
                   dataset,
                   input_ph,
                   label_ph,
                   minimize_op,
                   num_classes,
                   num_shots,
                   inner_batch_size,
                   inner_iters,
                   replacement,
                   meta_step_size,
                   meta_batch_size):
        pass

class HMAML(MetaLearner):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


def _sample_mini_dataset(dataset, num_classes, num_shots, task_type='classification'):
    """
    Sample a few shot task from a dataset.

    Returns:
      An iterable of (input, label) pairs.
    """

    if task_type == 'classification':
        shuffled = list(dataset)
        random.shuffle(shuffled)
        for class_idx, class_obj in enumerate(shuffled[:num_classes]):
            for sample in class_obj.sample(num_shots):
                yield (sample, class_idx)

    if task_type == 'regression':
        pass #sinewave = dataset.sample()




def _mini_batches(samples, batch_size, num_batches, replacement):
    """
    Generate mini-batches from some data.

    Returns:
      An iterable of sequences of (input, label) pairs,
        where each sequence is a mini-batch.
    """
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

def _split_train_test(samples, test_shots=1, task_type='classification'):
    """
    Split a few-shot task into a train and a test set.

    Args:
      samples: an iterable of (input, label) pairs.
      test_shots: the number of examples per class in the
        test set.

    Returns:
      A tuple (train, test), where train and test are
        sequences of (input, label) pairs.
    """
    train_set = list(samples)
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
