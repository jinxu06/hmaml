"""
Helpers for evaluating models.
"""

from components.meta_learners import MAML, FOML, Reptile, HMAML
from misc.variables import weight_decay

def evaluate(sess,
             model,
             dataset,
             num_classes=5,
             num_shots=5,
             eval_inner_batch_size=5,
             eval_inner_iters=50,
             replacement=False,
             num_samples=10000,
             transductive=False,
             weight_decay_rate=1,
             meta_learner=Reptile):
    """
    Evaluate a model on a dataset.
    """
    reptile = meta_learner(sess,
                         transductive=transductive,
                         pre_step_op=weight_decay(weight_decay_rate))
    total_correct = 0
    for _ in range(num_samples):
        total_correct += reptile.evaluate(dataset, model.input_ph, model.label_ph,
                                          model.minimize_op, model.predictions,
                                          num_classes=num_classes, num_shots=num_shots,
                                          inner_batch_size=eval_inner_batch_size,
                                          inner_iters=eval_inner_iters, replacement=replacement)
    return total_correct / (num_samples * num_classes)
