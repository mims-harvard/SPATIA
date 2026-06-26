
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.cli import ArgsType, SaveConfigCallback
from lightning.pytorch.loggers import WandbLogger
from scdataloader.datamodule_spatial import DataModule
from scprint.cli import MyCLI
from scprint.model.model_spatial import scPrint


class MySaveConfig(SaveConfigCallback):

    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        if type(trainer.logger) is WandbLogger:
            if self.config.get("wandblog", "") != "":
                trainer.logger.watch(
                    pl_module,
                    log=self.config.get("wandblog", "all"),
                    log_freq=self.config.get("wandblog_freq", 500),
                    log_graph=self.config.get("wandblog_graph", False),
                )
            if trainer.is_global_zero:
                print(trainer.datamodule)
                print(trainer.callbacks)
        return super().setup(trainer, pl_module, stage)


def main(args: ArgsType = None):
    try:
        cli = MyCLI(
            scPrint,
            DataModule,
            args=args,
            parser_kwargs={"parser_mode": "omegaconf"},
            save_config_kwargs={"overwrite": True},
            save_config_callback=MySaveConfig,
        )
    except Exception as e:
        import sys
        import traceback

        import ipdb

        exc_type, exc_value, tb = sys.exc_info()
        traceback.print_exc()
        ipdb.post_mortem(tb)


if __name__ == "__main__":
    main()
