import numpy as np
import tensorflow as tf
from misc.variables import (interpolate_vars, average_vars, subtract_vars,
                                    add_vars, scale_vars, VariableState)
import time

class Learner(object):

    def __init__(self, session, model, variables=None, transductive=False, pre_step_op=None):
        self.session = session
        self._model_state = VariableState(self.session, variables or tf.trainable_variables())
        self._full_state = VariableState(self.session,
                                         tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))
        self._transductive = transductive
        self._pre_step_op = pre_step_op

        self.model = model



    def evaluate(self, dataset, metrics=["loss", "accuracy"]):

        ops = [self.model.evals[m] for m in metrics]
        evals_sum = {m:0. for m in metrics}

        for count, (X, y) in enumerate(dataset):
            evals = self.session.run(ops, feed_dict={self.model.inputs:X, self.model.targets:y, self.model.sample_weights:np.ones((X.shape[0],))/X.shape[0], self.model.is_training:False})
            for i, m in enumerate(metrics):
                evals_sum[m] += evals[i]
        evals_mean = {m:evals_sum[m] / (count+1) for m in metrics}
        return evals_mean

    def train_step(self, minimize_op, feed_dict):
        _ = self.session.run(minimize_op, feed_dict=feed_dict)


    def one_shot_train_step(self, one_shot_data_iter, minimize_op, batch_size, step_size):
        begin = time.time()
        old_vars = self._model_state.export_variables()
        print(time.time()-begin)
        quit()
        updates = []
        for _ in range(batch_size):
            begin = time.time()
            X, y = next(one_shot_data_iter)
            end = time.time()
            print(end-begin)
            for i in range(2):
                last_vars = self._model_state.export_variables()
                _ = self.session.run(minimize_op, feed_dict={self.model.inputs:X, self.model.targets:y, self.model.is_training: True})
            end = time.time()
            print(end-begin)
            updates.append(subtract_vars(self._model_state.export_variables(), last_vars))
            end = time.time()
            print(end-begin)
            self._model_state.import_variables(old_vars)
            end = time.time()
            print(end-begin)
            print("......")
        update = average_vars(updates)
        self._model_state.import_variables(add_vars(old_vars, scale_vars(update, step_size)))


    # def train_step(self,
    #                dataset,
    #                input_ph,
    #                label_ph,
    #                minimize_op,
    #                num_classes,
    #                num_shots,
    #                inner_batch_size,
    #                inner_iters,
    #                replacement,
    #                meta_step_size,
    #                meta_batch_size):
    #     old_vars = self._model_state.export_variables()
    #     updates = []
    #     for _ in range(meta_batch_size):
    #         mini_dataset = dataset.sample_task(num_shots, num_classes)
    #         mini_batches = dataset.mini_batch(mini_dataset, inner_batch_size, inner_iters,
    #                                           replacement)
    #         for batch in mini_batches:
    #             inputs, labels = zip(*batch)
    #             last_backup = self._model_state.export_variables()
    #             if self._pre_step_op:
    #                 self.session.run(self._pre_step_op)
    #             self.session.run(minimize_op, feed_dict={input_ph: inputs, label_ph: labels})
    #         updates.append(subtract_vars(self._model_state.export_variables(), last_backup))
    #         self._model_state.import_variables(old_vars)
    #     update = average_vars(updates)
    #     self._model_state.import_variables(add_vars(old_vars, scale_vars(update, meta_step_size)))


    # def evaluate(self,
    #              dataset,
    #              input_ph,
    #              label_ph,
    #              minimize_op,
    #              predictions,
    #              num_classes,
    #              num_shots,
    #              inner_batch_size,
    #              inner_iters,
    #              replacement):
    #     # train_set, test_set = _split_train_test(
    #     #     _sample_mini_dataset(dataset, num_classes, num_shots+1))
    #     train_set, test_set = dataset.sample_task(num_shots+1, num_classes, use_split=True, test_shots=1)
    #     old_vars = self._full_state.export_variables()
    #     mini_batches = dataset.mini_batch(train_set, inner_batch_size, inner_iters, replacement)
    #     for batch in mini_batches:
    #         inputs, labels = zip(*batch)
    #         if self._pre_step_op:
    #             self.session.run(self._pre_step_op)
    #         self.session.run(minimize_op, feed_dict={input_ph: inputs, label_ph: labels})
    #     test_preds = self._test_predictions(train_set, test_set, input_ph, predictions)
    #     #num_correct = 0 #sum([pred == sample[1] for pred, sample in zip(test_preds, test_set)])
    #     e = self.eval_metric(test_preds, labels)
    #     self._full_state.import_variables(old_vars)
    #     return e
    #
    # def _test_predictions(self, train_set, test_set, input_ph, predictions):
    #     if self._transductive:
    #         inputs, _ = zip(*test_set)
    #         return self.session.run(predictions, feed_dict={input_ph: inputs})
    #     res = []
    #     for test_sample in test_set:
    #         inputs, _ = zip(*train_set)
    #         inputs += (test_sample[0],)
    #         res.append(self.session.run(predictions, feed_dict={input_ph: inputs})[-1])
    #     return res
