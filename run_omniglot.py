"""
Train a model on Omniglot.
"""
import os
import random
import tensorflow as tf

from misc.args import argument_parser, model_kwargs, train_kwargs, evaluate_kwargs
from executions.eval import evaluate
from components.models import OmniglotModel
from data.omniglot import read_dataset, split_dataset, augment_dataset
from executions.train import train
from data.omniglot import OmniglotDataSource
from data.load_data import Dataset

DATA_DIR = "/Users/Aaron-MAC/Code/supervised-reptile/data/omniglot"
CHECKPOINT_DIR = "model_checkpoint"
# DATA_DIR = "/data/ziz/not-backed-up/jxu/omniglot"
# CHECKPOINT_DIR = "/data/ziz/jxu/hmaml-checkpoints"


def main():
    """
    Load data and train a model on it.
    """
    args = argument_parser().parse_args()
    random.seed(args.seed)

    args.checkpoint_dir = CHECKPOINT_DIR ###
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus

    data_source = OmniglotDataSource(data_dir=DATA_DIR)
    data_source.split_train_test(num_train=1200)
    train_set = Dataset(data_source, which_set='train', task_type='classification')
    test_set = Dataset(data_source, which_set='test', task_type='classification')

    model = OmniglotModel(args.classes, **model_kwargs(args))

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        if not args.pretrained:
            print('Training...')
            train(sess, model, train_set, test_set, os.path.join(args.checkpoint_dir, args.checkpoint), **train_kwargs(args))
        else:
            print('Restoring from checkpoint...')
            tf.train.Saver().restore(sess, tf.train.latest_checkpoint(os.path.join(args.checkpoint_dir, args.checkpoint)))

        print('Evaluating...')
        eval_kwargs = evaluate_kwargs(args)
        print('Train accuracy: ' + str(evaluate(sess, model, train_set, **eval_kwargs)))
        print('Test accuracy: ' + str(evaluate(sess, model, test_set, **eval_kwargs)))

if __name__ == '__main__':
    main()
