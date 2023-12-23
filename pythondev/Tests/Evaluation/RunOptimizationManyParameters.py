from Configurations import *
from Tests.Utils.TestsUtils import DB

from Tests.Evaluation.ParameterOptimizationTool import get_args, optimize_parameter


def nested_dictionary_to_tuple_key(d: dict):
    result = {}
    for key, val in d.items():
        if isinstance(val, dict):
            result.update({(key,) + x: v for x, v in nested_dictionary_to_tuple_key(val).items()})
        else:
            result[(key,)] = val
    return result


params = {'rr':
              {'reliability_params': {
                                     'no_reliable_time_to_stop_high_quality': [5, 10],  # sec
                     }}}
original_tupled = nested_dictionary_to_tuple_key(back_chair_config)

for key, val in nested_dictionary_to_tuple_key(params).items():
    args = get_args()
    args.parameter_name = key[-1]
    args.values = val
    args.original_value = original_tupled[key]
    args.version = key[0] + '_' + key[-1]
    optimize_parameter(args, DB())
