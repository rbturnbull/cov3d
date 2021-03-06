import typer
from pathlib import Path
from sklearn.metrics import f1_score
import pandas as pd
from cov3d.apps import Covideo, CovideoSeverity
import wandb

def main(model_dir:Path, mc_samples:int=0, log_wandb:bool = True, mc_dropout:bool=False, severity:bool=False):
    app = Covideo() if not severity else CovideoSeverity()
    covid_results = app(
        pretrained=model_dir/"export.pkl",
        scan_dir="../validation/covid/",
        # scan=[Path("../validation/covid/ct_scan_156")],
        # scan_dir=[], # hack
        scan=[],
        mc_samples=mc_samples,
        output_csv=model_dir/"validation-covid.csv",
        output_mc=model_dir/"validation-covid.pt",
        noncovid_txt=model_dir/"validation.fn.txt",
        covid_txt=model_dir/"validation.tp.txt",
        mild_txt=model_dir/"mild.txt",
        moderate_txt=model_dir/"moderate.txt",
        severe_txt=model_dir/"severe.txt",
        critical_txt=model_dir/"critical.txt",
        mc_dropout=mc_dropout,
    )
    covid_results["true"] = 1

    severity_df = pd.read_csv("../val_partition_covid_categories.csv", sep=";")
    severity_categories = ["mild", "moderate", "severe", "critical"]
    covid_results["true_severity"] = "-"
    for _, row in severity_df.iterrows():
        covid_results.loc[ covid_results["name"] == row['Name'], 'true_severity' ] = severity_categories[int(row['Category']) - 1]

    if not severity:
        noncovid_results = app(
            pretrained=model_dir/"export.pkl",
            scan_dir="../validation/non-covid/",
            output_csv=model_dir/"validation-noncovid.csv",
            output_mc=model_dir/"validation-noncovid.pt",
            # scan="../validation/non-covid/ct_scan_3",
            # scan_dir=[], # hack
            scan=[],
            mc_samples=mc_samples,
            noncovid_txt=model_dir/"validation.tn.txt",
            covid_txt=model_dir/"validation.fp.txt",
            mc_dropout=mc_dropout,
        )
        noncovid_results["true"] = 0
        results = pd.concat([covid_results,noncovid_results])
    else:
        results = covid_results
        
    macro_f1 = f1_score(results["true"], results["COVID19 positive"], average="macro")
    results.to_csv(model_dir/"validation.csv", index=False)
    print("macro_f1", macro_f1)
    # binary_f1 = f1_score(results["true"], results["COVID19 positive"])
    # print("binary_f1", binary_f1)

    severity_results = covid_results[ covid_results["true_severity"] != "-" ]
    severity_f1 = f1_score(severity_results["true_severity"], severity_results["severity"], average="macro")

    print("severity_f1", severity_f1)

    with open(model_dir/"validation.macrof1.txt", 'w') as f:
        print('presence', macro_f1, file=f, sep=',')
        print('severity', severity_f1, file=f, sep=',')

    if log_wandb:
        with wandb.init(id=model_dir.name, project=app.project_name(), resume="must") as run:
            wandb.log({"presence_f1_final": macro_f1, "severity_f1_final": severity_f1})


if __name__ == "__main__":
    typer.run(main)
