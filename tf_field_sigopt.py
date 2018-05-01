import json, argparse

from sigopt import Connection
from sigopt_objective import calculate_objective
import numpy as np

parser = argparse.ArgumentParser(description='Experiment ID')###DO NOT PUT THE NAME HERE

parser.add_argument('--experiment-id', type=int)

args = parser.parse_args()
experiment_id = 44843 #args.experiment_id

# Instantiate Connection Object
SIGOPT_API_TOKEN = 'WVLHORZUADZXFGWVPXFLRDQNBDSWBOVIFKTZPUHZAWTDLPZY'
conn = Connection(client_token=SIGOPT_API_TOKEN)

# Get experiment object
if experiment_id is None:

    # Get hyperparameters
    exp_name = 'ultrasound_image_alignment'   ## ID 44829  HARDCODED ABOVE
    param_filepath='hyperparams.json'

    with open(param_filepath) as f:
        hyperparams = f.read()
        hyperparams = json.loads(hyperparams)

    experiment = conn.experiments().create(
                         name=exp_name,
                         parameters=hyperparams,
                         observation_budget=20*len(hyperparams),
                         metrics =  [{'name': 'mse'},
                                     {'name': 'runtime'}],
                        linear_constraints=[
                            # Constraint equation: turn_on_elastic_frac - turn_on_rotation_frac > 0
                            dict(
                                type='greater_than',
                                threshold=0,
                                terms=[
                                    dict(name='turn_on_rotation_frac', weight=-1),
                                    dict(name='turn_on_elastic_frac', weight=1),
                                ],
                            ),
                        ],
    )
    print("Created experiment: https://sigopt.com/experiment/" + experiment.id)
else:
    experiment = conn.experiments(experiment_id).fetch()

# Optimization Loop

while experiment.progress.observation_count < experiment.observation_budget:
    suggestion = conn.experiments(experiment.id).suggestions().create()

    try:
        values, mse_loss = calculate_objective(suggestion.assignments, suggestion.id)
        print("Values are {}".format(values))
        print("mse_loss is {}".format(mse_loss))
        if values is not np.nan:
            observation = conn.experiments(experiment.id).observations().create(
                values=values,
                suggestion=suggestion.id,
                metadata=dict(
                    mse_loss=mse_loss
                )
            )
        else:
            observation = conn.experiments(experiment.id).observations().create(
                failed=True,
                suggestion=suggestion.id)
    except:
        observation = conn.experiments(experiment.id).observations().create(
            failed=True,
            suggestion=suggestion.id)
