import numpy as np
import tensorflow as tf
import collections
from tensorflow_field import align_images


def calculate_objective(assignments):

    tf.reset_default_graph()

    hpms = collections.namedtuple('hparams',['name', 'num_steps', 'learning_rate', 'beta', 'initial_scale','scale_tuner_alpha', 'coherence_weight' ])

    hparams_run = hpms(name='test',
                       num_steps = assignments['num_steps'],
                       learning_rate= assignments['learning_rate'],
                       beta=assignments['beta'],
                       initial_scale=assignments['initial_scale'],
                       scale_tuner_alpha= assignments['scale_tuner_alpha'],
                       coherence_weight= assignments['coherence_weight'])

    try:
        metric = align_images(hparams_run.name, hparams_run)
        objectives = [{'name': 'loss', 'value': float(metric)}]
        return objectives
    except:
        return np.nan



if __name__ == "__main__":
    assignments= {
                  'num_epochs': .5,
                  'batch_size':34,
                  'log_learning_rate':-8,
                  'log_one_minus_beta':-1,
                  'dropout_rate':.9,
                  'fc_num':500
                  }

    calculate_objective(assignments)