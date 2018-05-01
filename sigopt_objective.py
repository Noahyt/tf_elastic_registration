import numpy as np
import tensorflow as tf
import collections
from tensorflow_field import align_images


def calculate_objective(assignments, id):
    tf.reset_default_graph()

    hpms = collections.namedtuple('hparams',
                                  ['name', 'num_steps', 'learning_rate', 'beta', 'initial_scale', 'scale_tuner_alpha',
                                   'elastic_weight', 'translation_coherence_weight', 'rotation_coherence_weight',
                                   'turn_on_rotation_frac', 'turn_on_elastic_frac', ])

    hparams_run = hpms(name=id,
                       num_steps=assignments['num_steps'],
                       learning_rate=assignments['learning_rate'],
                       beta=assignments['beta'],
                       initial_scale=assignments['initial_scale'],
                       scale_tuner_alpha=assignments['scale_tuner_alpha'],
                       elastic_weight=assignments['elastic_weight'],
                       translation_coherence_weight=assignments['translation_coherence_weight'],
                       rotation_coherence_weight=assignments['rotation_coherence_weight'],
                       turn_on_rotation_frac=assignments['turn_on_rotation_frac'],
                       turn_on_elastic_frac=assignments['turn_on_elastic_frac']
                       )

    print(hparams_run)

    try:
        metric, runtime, mse_loss = align_images(directory='output/' + hparams_run.name, hparams=hparams_run, save_figs=True)
        if metric is np.nan:
            return np.nan, np.nan
        objectives = [{'name': 'mse', 'value': float(-metric)}, {'name': 'runtime', 'value': float(-runtime)}]
        return objectives, mse_loss
    except:
        return np.nan, np.nan, np.nan
