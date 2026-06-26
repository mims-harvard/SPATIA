import os

import pandas as pd
from scprint.utils.utils import run_command
from torch import load


class PROTBERT:
    def __init__(
        self,
        config: str = "esm-extract",
        pretrained_model: str = "esm2_t33_650M_UR50D",
    ):
        self.config = config
        self.pretrained_model = pretrained_model

    def __call__(
        self, input_file: str, output_folder: str = "/tmp/esm_out/", cache: bool = True
    ) -> pd.DataFrame:
        if not os.path.exists(output_folder) or not cache:
            os.makedirs(output_folder, exist_ok=True)
            cmd = (
                self.config
                + " "
                + self.pretrained_model
                + " "
                + input_file
                + " "
                + output_folder
                + " --include mean"
            )
            print(f"running protbert command: {cmd}")
            try:
                run_command(cmd, shell=True)
            except Exception as e:
                raise RuntimeError(
                    "An error occurred while running the esm-extract command: " + str(e)
                )
        return self.read_results(output_folder)

    def read_results(self, output_folder):
        files = os.listdir(output_folder)
        files = [i for i in files if i.endswith(".pt")]
        results = []
        for file in files:
            results.append(
                load(output_folder + file)["mean_representations"][33].numpy().tolist()
            )
        return pd.DataFrame(data=results, index=[file.split(".")[0] for file in files])
