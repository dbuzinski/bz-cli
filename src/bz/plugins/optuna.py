from optuna.distributions import json_to_distribution
from optuna import create_study

from .plugin import Plugin


class OptunaPlugin(Plugin):
    def __init__(self, optuna_config, study_config):
        self.study = create_study(**study_config)
        self.param_specs = optuna_config
        self.trial = None
        self.objective = "loss"
        self.params = {}

    def start_training_session(self, context):
        # Ask for a new trial
        self.trial = self.study.ask()
        # Generate param values
        self.params = {
            name: self.trial._suggest(name, json_to_distribution(dist))
            for name, dist in self.param_specs.items()
        }
        # Inject into config
        context.config.update(self.params)

    def end_validation_loop(self, context):
        objective = context.validation_metrics.get(self.objective)
        if objective is not None:
            self.study.tell(self.trial, objective)

    def end_training_session(self, context):
        self.study.stop()

    @staticmethod
    def init(spec):
        optuna_config = load_json("optuna_config.json")
        study_config = spec.config.get("optuna", {})
        return OptunaPlugin(optuna_config, study_config)
