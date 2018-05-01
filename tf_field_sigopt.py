import json, argparse

from sigopt import Connection
from sigopt_objective import calculate_objective
import numpy as np

parser = argparse.ArgumentParser(description='Experiment ID')###DO NOT PUT THE NAME HERE

parser.add_argument('--experiment-id', type=int)

args = parser.parse_args()
experiment_id = args.experiment_id

# Instantiate Connection Object
SIGOPT_API_TOKEN = 'WVLHORZUADZXFGWVPXFLRDQNBDSWBOVIFKTZPUHZAWTDLPZY'
conn = Connection(client_token=SIGOPT_API_TOKEN)

# Get experiment object
if experiment_id is None:

    # Get hyperparameters
    exp_name = 'ultrasound_image_alignment'   ## ID 34928  HARDCODED ABOVE
    param_filepath='hyperparams.json'

    with open(param_filepath) as f:
        hyperparams = f.read()
        hyperparams = json.loads(hyperparams)

    experiment = conn.experiments().create(
                         name=exp_name,
                         parameters=hyperparams,
                         observation_budget=20*len(hyperparams),
                         metrics =  [{'name': 'r2_value'}]
    )
    print("Created experiment: https://sigopt.com/experiment/" + experiment.id)
else:
    experiment = conn.experiments(experiment_id).fetch()

# Optimization Loop

while experiment.progress.observation_count < experiment.observation_budget:
    suggestion = conn.experiments(experiment.id).suggestions().create()

    try:
        value = calculate_objective(suggestion.assignments)
        print("Value is {}".format(value))

        if value is not np.nan:
            observation = conn.experiments(experiment.id).observations().create(
                values=value,
                suggestion=suggestion.id)
        else:
            observation = conn.experiments(experiment.id).observations().create(
                failed=True,
                suggestion=suggestion.id)
    except:
        observation = conn.experiments(experiment.id).observations().create(
            failed=True,
            suggestion=suggestion.id)
