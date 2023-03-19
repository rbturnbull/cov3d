import re
import pandas as pd
import typer
from sklearn.metrics import f1_score
from typing import List
from pathlib import Path
from cov3d.apps import Cov3d

def main(output_dir:Path, model_dirs:List[Path]):
    for model_dir in model_dirs:
        validation_log = model_dir/"validation.macrof1.txt"
        print('--'*4)
        print(model_dir.name)
        print(validation_log.read_text())
        print('--')
        
    def get_df(filename):
        return pd.concat([pd.read_csv(model_dir/filename) for model_dir in model_dirs])

    covid_df = get_df("validation-covid.csv")
    covid_df_averaged = covid_df.groupby(['path'])[['probability']].mean()
    covid_df_averaged["true"] = True

    noncovid_df = get_df("validation-noncovid.csv")
    noncovid_df_averaged = noncovid_df.groupby(['path'])[['probability']].mean()
    noncovid_df_averaged["true"] = False

    task1_df = pd.concat([covid_df_averaged, noncovid_df_averaged])
    task1_df["COVID19 positive"] = task1_df['probability'] >= 0.5
    task1_macro_f1 = f1_score(task1_df["true"], task1_df["COVID19 positive"], average="macro")
    task1_f1_raw = f1_score(task1_df["true"], task1_df["COVID19 positive"], average=None)

    task1_df.to_csv(output_dir/"validation-presence.csv")

    print("ENSEMBLE")
    model_list_file = output_dir/"models.validation.txt"
    model_names = [model_path.name for model_path in model_dirs]
    model_list_file.write_text("\n".join(model_names)+"\n")
    output_dir.mkdir(exist_ok=True, parents=True)
    print(f"Models list: {model_list_file}")

    print("task1 macro_f1", task1_macro_f1)
    # print("task1 f1_raw", task1_f1_raw)

    severity_df = get_df("severity.csv")
    severity_df_averaged = severity_df.groupby(['path'])[['mild_probability', 'moderate_probability', 'severe_probability', 'critical_probability']].mean()
    severity_df_averaged["severity"] = severity_df_averaged[['mild_probability', 'moderate_probability', 'severe_probability', 'critical_probability']].idxmax(axis=1).str.replace("_probability","")
    severity_df_averaged = severity_df_averaged.reset_index()
    severity_df_averaged["path"] = severity_df_averaged["path"].str.replace(r"^\.\.\/","", regex=True).astype(str)
    # severity_df_averaged = severity_df_averaged.sort_values(by="path")


    # severity_df_averaged = severity_df_averaged.sort_values(by=['col1'])

    df1 = pd.read_csv("cross-validation.csv")
    cross_validation_df = pd.read_csv("cross-validation.csv")
    cross_validation_df["path"] = cross_validation_df["path"].astype(str)

    cross_validation_df = cross_validation_df[cross_validation_df.path.isin(severity_df_averaged.path)]
    severity_df_averaged = severity_df_averaged.join(cross_validation_df[["path", "category"]].set_index('path'), on="path")
    
    task2_macro_f1 = f1_score(severity_df_averaged["category"].str.lower(), severity_df_averaged["severity"].str.lower(), average="macro")

    severity_df_averaged.to_csv(output_dir/"validation-severity.csv")
    print('task2_macro_f1', task2_macro_f1)

    with open(output_dir/"validation.macrof1.txt", 'w') as f:
        print('presence', task1_macro_f1, file=f, sep=',')
        print('severity', task2_macro_f1, file=f, sep=',')





if __name__ == "__main__":
    typer.run(main)
