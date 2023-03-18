import pandas as pd
import typer
from typing import List
from pathlib import Path
from cov3d.apps import Cov3d

def main(output_dir:Path, model_dirs:List[Path]):
    presence_predictions_dfs = [pd.read_csv(model_dir/"presence-test-predictions.csv") for model_dir in model_dirs]
    presence_predictions = pd.concat(presence_predictions_dfs)

    output_dir.mkdir(exist_ok=True, parents=True)
    noncovid_txt=output_dir/"non-covid.csv",
    covid_txt=output_dir/"covid.csv",



if __name__ == "__main__":
    typer.run(main)
