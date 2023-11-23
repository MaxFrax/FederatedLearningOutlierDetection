import logging

LOGGER = logging.getLogger(__name__)

def tune_model(model):
    LOGGER.info('Tuning model')

    model.setParam('TuneTimeLimit', 500)
    model.setParam('TuneTrials', 3)
    model.setParam('TuneCriterion', 2)
    model.setParam('TuneResults', 3)
    model.setParam('TuneOutput', 1)

    model.tune()
    LOGGER.info('Tuning done')

    for i in range(model.tuneResultCount):
        model.getTuneResult(i)
        model.write(f'{model.__name__}{i}.prm')

    model.getTuneResult(0)

def error_code_to_string(code):
    codes = {
                    1: "OPTIMAL",
                    2: "INFEASIBLE",
                    3: "INF_OR_UNBD",
                    4: "INFEASIBLE_OR_UNBOUNDED",
                    5: "UNBOUNDED",
                    6: "CUTOFF",
                    7: "ITERATION_LIMIT",
                    8: "NODE_LIMIT",
                    9: "TIME_LIMIT",
                    10: "SOLUTION_LIMIT",
                    11: "INTERRUPTED",
                    12: "NUMERIC",
                    13: "SUBOPTIMAL",
                    14: "INPROGRESS",
                    15: "USER_OBJ_LIMIT"
                }
    return codes[code]