"""In-memory logger -- when you want to access your learning trajectory directly after
learning.
"""

import torch

from pytorch_lightning.loggers.base import LightningLoggerBase, rank_zero_experiment
from pytorch_lightning.utilities.logger import _add_prefix, _convert_params
from pytorch_lightning.utilities.rank_zero import rank_zero_only

from argparse import Namespace
from typing import Optional, Union, List, Dict, Any

import pandas as pd


class ExperimentWriter:
    """In-memory experiment writer.

    Currently this supports logging hyperparameters and metrics.
    """

    def __init__(self):
        self.hparams: Dict[str, Any] = {}
        self.metrics: List[Dict[str, float]] = []

    def log_hparams(self, params: Dict[str, Any]):
        """Record hyperparameters.

        This adds to previously recorded hparams, overwriting exisiting values in case
        of repeated keys.
        """
        self.hparams.update(params)

    def log_metrics(self, metrics_dict: Dict[str, float], step: Optional[int] = None):
        """Record metrics."""

        def _handle_value(value: Union[torch.Tensor, Any]) -> Any:
            if isinstance(value, torch.Tensor):
                return value.item()
            return value

        if step is None:
            step = len(self.metrics)

        metrics = {k: _handle_value(v) for k, v in metrics_dict.items()}
        metrics["step"] = step
        self.metrics.append(metrics)


class InMemoryLogger(LightningLoggerBase):
    """Log to in-memory storage.

    :param name: experiment name
    :param version: experiment version
    :param prefix: string to put at the beginning of metric keys
    :param df_aggregate_by_step: if true, the metrics dataframe is aggregated by step,
        with repeated non-NA entries averaged over
    :param hparams: access recorded hyperparameters
    :param metrics: access recorded metrics
    :param metrics_df: access metrics in Pandas dataframe format; this is only available
        after `finalize` or `save`
    """

    LOGGER_JOIN_CHAR = "-"

    def __init__(
        self,
        name: str = "in_memory",
        version: int = 1,
        prefix: str = "",
        df_aggregate_by_step: bool = True,
    ):
        super().__init__()
        self._name = name
        self._version = version
        self._prefix = prefix
        self._metrics_df = None

        self.df_aggregate_by_step = df_aggregate_by_step

        self._experiment: Optional[ExperimentWriter] = None

    @property
    @rank_zero_experiment
    def experiment(self) -> ExperimentWriter:
        """Access the actual logger object."""
        if self._experiment is None:
            self._experiment = ExperimentWriter()

        return self._experiment

    @rank_zero_only
    def log_hyperparams(self, params: Union[Dict[str, Any], Namespace]):
        """Log hyperparameters."""
        params = _convert_params(params)
        self.experiment.log_hparams(params)

    @rank_zero_only
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log metrics, associating with given `step`."""
        metrics = _add_prefix(metrics, self._prefix, self.LOGGER_JOIN_CHAR)
        self.experiment.log_metrics(metrics, step)

    def pandas(self):
        """Return recorded metrics in a Pandas dataframe.

        By default the dataframe is aggregated by step, with repeated non-NA entries
        averaged over. This can be disabled using `self.df_aggregate_by_step`.
        """
        df = pd.DataFrame(self.experiment.metrics)
        if len(df) > 0:
            df.set_index("step", inplace=True)
            if self.df_aggregate_by_step:
                df = df.groupby("step").mean()

        return df

    def save(self):
        """Generate the metrics dataframe."""
        self._metrics_df = self.pandas()

    @property
    def name(self) -> str:
        """Experiment name."""
        return self._name

    @property
    def version(self) -> int:
        """Experiment version. Not used by this lgoger."""
        return self._version

    @property
    def hparams(self) -> Dict[str, Any]:
        """Access recorded hyperparameters.

        This is equivalent to `self.experiment.hparams`.
        """
        assert self._experiment is not None
        return self._experiment.hparams

    @property
    def metrics(self) -> List[Dict[str, float]]:
        """Access recorded metrics.

        This is equivalent to `self.experiment.metrics`.
        """
        assert self._experiment is not None
        return self._experiment.metrics

    @property
    def metrics_df(self) -> pd.DataFrame:
        """Access recorded metrics in Pandas format.

        This is a cached version of the output from `self.pandas()` generated on
        `self.save()` or `self.finalize()`.
        """
        return self._metrics_df
