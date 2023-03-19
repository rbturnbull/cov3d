import re
import pandas as pd
import typer
from typing import List
from pathlib import Path
from cov3d.apps import Cov3d

def main(output_dir:Path, model_dirs:List[Path]):
    presence_predictions_dfs = [pd.read_csv(model_dir/"test-predictions/presence-test-predictions.csv") for model_dir in model_dirs]
    presence_predictions = pd.concat(presence_predictions_dfs)

    severity_predictions_dfs = [pd.read_csv(model_dir/"test-predictions/severity-test-predictions.csv") for model_dir in model_dirs]
    severity_predictions = pd.concat(severity_predictions_dfs)


    model_list_file = output_dir/"models.txt"
    model_names = [model_path.name for model_path in model_dirs]
    model_list_file.write_text("\n".join(model_names)+"\n")
    output_dir.mkdir(exist_ok=True, parents=True)
    print(f"Models list: {model_list_file}")

    presence_predictions_averaged = presence_predictions.groupby(['name'])[['probability']].mean()
    severity_predictions_averaged = severity_predictions.groupby(['name'])[['mild_probability', 'moderate_probability', 'severe_probability', 'critical_probability']].mean()

    presence_predictions_averaged["COVID19 positive"] = presence_predictions_averaged['probability'] >= 0.5
    severity_predictions_averaged["severity"] = severity_predictions_averaged[['mild_probability', 'moderate_probability', 'severe_probability', 'critical_probability']].idxmax(axis=1).str.replace("_probability","")
    presence_predictions_averaged["name"] = presence_predictions_averaged.index
    severity_predictions_averaged["name"] = severity_predictions_averaged.index

    def get_digits(string):
        m = re.search(r"\d+", string)
        if m:
            return int(m.group(0))
        return -1

    def write_scans_txt(filename, df, mask):
        if filename:
            scans = df[mask]["name"].tolist()
            scans = sorted(scans, key=get_digits)
            print(f"writing to {filename} ({len(scans)} scans)")
            with open(filename, "w") as f:
                f.write("\n".join(scans) + "\n")

    presence_predictions_averaged.to_csv(output_dir/"presence-test-predictions.csv")
    severity_predictions_averaged.to_csv(output_dir/"severity-test-predictions.csv")
    write_scans_txt(output_dir/"covid.csv", presence_predictions_averaged, presence_predictions_averaged["COVID19 positive"] == True)
    write_scans_txt(output_dir/"non-covid.csv", presence_predictions_averaged, presence_predictions_averaged["COVID19 positive"] == False)

    write_scans_txt(output_dir/"mild.csv", severity_predictions_averaged, severity_predictions_averaged["severity"] == "mild")
    write_scans_txt(output_dir/"moderate.csv", severity_predictions_averaged, severity_predictions_averaged["severity"] == "moderate")
    write_scans_txt(output_dir/"severe.csv", severity_predictions_averaged, severity_predictions_averaged["severity"] == "severe")
    write_scans_txt(output_dir/"critical.csv", severity_predictions_averaged, severity_predictions_averaged["severity"] == "critical")



if __name__ == "__main__":
    typer.run(main)
