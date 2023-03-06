import typer
from pathlib import Path
import pandas as pd
import numpy as np
import random

def main(
    directory: Path,
    training_severity: Path = None,
    validation_severity: Path = None,
    extra_splits:int=4,
    seed:int=42,
):
    random.seed(seed)

    severity_categories = [
        "-",
        "Mild",
        "Moderate",
        "Severe",
        "Critical",
    ]

    training_severity = training_severity or directory/"ICASSP_severity_train_partition.xlsx"
    validation_severity = validation_severity or directory/"ICASSP_severity_validation_partition.xlsx"
    training_severity_df = pd.read_excel(training_severity)
    validation_severity_df = pd.read_excel(validation_severity)

    print("path,split,has_covid,category")
    split = 0
    has_covid = 1
    for _, row in validation_severity_df.iterrows():
        path = Path("validation", "covid", row["Name"])
        severity = severity_categories[row["Category"]]
        print(path,split,has_covid,severity, sep=",")        

    training_severity_df = pd.read_excel(training_severity)

    for category in training_severity_df.Category.unique():
        category_list = training_severity_df[training_severity_df.Category == category]["Name"].tolist()
        random.shuffle(category_list)
        category_splits = np.array_split(category_list, extra_splits)
        for i, names in enumerate(category_splits):
            for name in names:
                path = Path("train", "covid", name)
                split = 1 + i
                row = training_severity_df[training_severity_df.Name == name]
                if len(row) > 1:
                    row = row.iloc[0]
                severity = severity_categories[row["Category"].item()]

                print(path,split,has_covid,severity, sep=",")

    def get_scans(subdir:Path):
        scans = [path for path in subdir.iterdir() if path.name.startswith("ct_scan")]
        random.shuffle(scans)
        return scans

    # Add validation non-covid
    split = 0
    has_covid = 0
    severity = "non-covid"
    for folder in get_scans(directory/"validation"/"non-covid"):
        path = Path("validation", "non-covid", folder.name)
        print(path,split,has_covid,severity, sep=",")

    # Add training non-covid
    has_covid = 0
    severity = "non-covid"
    category_splits = np.array_split(get_scans(directory/"train"/"non-covid"), extra_splits)
    for i, names in enumerate(category_splits):
        for name in names:
            path = Path("train", "non-covid", name.name)
            split = 1 + i
            print(path,split,has_covid,severity, sep=",")

    # Add validation covid (excluding severity listing)
    scans = [scan for scan in get_scans(directory/"validation"/"covid") if (validation_severity_df.Name == scan.name).sum() == 0]
    has_covid = 1
    severity = "covid"
    split = 0
    category_splits = np.array_split(scans, extra_splits)
    for scan in scans:
        path = Path("validation", "covid", scan.name)
        print(path,split,has_covid,severity, sep=",")

    # Add training covid (excluding severity listing)
    scans = [scan for scan in get_scans(directory/"train"/"covid") if (training_severity_df.Name == scan.name).sum() == 0]
    has_covid = 1
    severity = "covid"
    category_splits = np.array_split(scans, extra_splits)
    for i, names in enumerate(category_splits):
        for name in names:
            path = Path("train", "covid", name.name)
            split = 1 + i
            print(path,split,has_covid,severity, sep=",")



if __name__ == "__main__":
    typer.run(main)
