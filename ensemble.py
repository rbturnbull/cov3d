import re
import pandas as pd
import typer
from typing import List
from pathlib import Path
from cov3d.apps import Cov3d

def main(output_dir:Path, model_dirs:List[Path]):
    presence_predictions_dfs = [pd.read_csv(model_dir/"test-predictions/presence-test-predictions.csv") for model_dir in model_dirs]
    presence_predictions = pd.concat(presence_predictions_dfs)

    averaged = presence_predictions.groupby(['name'])[['probability', 'mild_probability', 'moderate_probability', 'severe_probability', 'critical_probability']].mean()

    averaged["COVID19 positive"] = averaged['probability'] >= 0.5
    averaged["severity"] = averaged[['mild_probability', 'moderate_probability', 'severe_probability', 'critical_probability']].idxmax(axis=1).str.replace("_probability","")
    averaged["name"] = averaged.index
    output_dir.mkdir(exist_ok=True, parents=True)

    def get_digits(string):
        m = re.search(r"\d+", string)
        if m:
            return int(m.group(0))
        return -1

    def write_scans_txt(filename, mask):
        if filename:
            scans = averaged[mask]["name"].tolist()
            scans = sorted(scans, key=get_digits)
            print(f"writing to {filename} ({len(scans)} scans)")
            with open(filename, "w") as f:
                f.write("\n".join(scans) + "\n")

    model_list_file = output_dir/"models.txt"
    model_names = [model_path.name for model_path in model_dirs]
    model_list_file.write_text("\n".join(model_names)+"\n")
    print(f"Models list: {model_list_file}")
    write_scans_txt(output_dir/"covid.csv", averaged["COVID19 positive"] == True)
    write_scans_txt(output_dir/"non-covid.csv", averaged["COVID19 positive"] == False)
    write_scans_txt(output_dir/"mild.csv", averaged["severity"] == "mild")
    write_scans_txt(output_dir/"moderate.csv", averaged["severity"] == "moderate")
    write_scans_txt(output_dir/"severe.csv", averaged["severity"] == "severe")
    write_scans_txt(output_dir/"critical.csv", averaged["severity"] == "critical")



if __name__ == "__main__":
    typer.run(main)
