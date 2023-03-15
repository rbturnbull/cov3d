import typer
from pathlib import Path
from sklearn.metrics import f1_score
import pandas as pd
from cov3d.apps import Cov3d
import wandb

def main(model_dir:Path, split:int=0, mc_samples:int=0, log_wandb:bool = False, mc_dropout:bool=False):
    app = Cov3d()

    cross_validation_df = pd.read_csv("cross-validation.csv")
    split_df = cross_validation_df[ cross_validation_df.split == split ]
    has_covid_df = split_df[ split_df.has_covid == True].reset_index()

    covid_results = app(
        pretrained=model_dir/"presence_f1.best.pkl",
        # scan_dir="../validation/covid/",
        # scan=[Path("../validation/covid/ct_scan_156")],
        # scan_dir=[], # hack
        scan=[Path(x) for x in has_covid_df.path],
        mc_samples=mc_samples,
        directory=Path(".."),
        output_csv=model_dir/"validation-covid.csv",
        output_mc=model_dir/"validation-covid.pt",
        noncovid_txt=model_dir/"validation.fn.txt",
        covid_txt=model_dir/"validation.tp.txt",
        mc_dropout=mc_dropout,
    )
    covid_results["true"] = 1
    covid_results["true_severity"] = has_covid_df["category"].replace("covid","-")

    severity_df = has_covid_df[has_covid_df.category.str.lower() != "covid"]

    severity_columns = ['severity', 'mild_samples', 'moderate_samples', 'severe_samples',
       'critical_samples', 'mild_probability', 'moderate_probability',
       'severe_probability', 'critical_probability',
    ]
    covid_results = covid_results.drop(columns=severity_columns)

    severity_results = app(
        pretrained=model_dir/"severity_f1.best.pkl",
        # scan_dir="../validation/covid/",
        # scan=[Path("../validation/covid/ct_scan_156")],
        # scan_dir=[], # hack
        scan=[Path(x) for x in severity_df.path],
        mc_samples=mc_samples,
        directory=Path(".."),
        output_csv=model_dir/"severity.csv",
        output_mc=model_dir/"validation-severity.pt",
        mild_txt=model_dir/"mild.txt",
        moderate_txt=model_dir/"moderate.txt",
        severe_txt=model_dir/"severe.txt",
        critical_txt=model_dir/"critical.txt",
        mc_dropout=mc_dropout,
    )    
    severity_results.path = severity_results.path.astype(str)
    covid_results.path = covid_results.path.astype(str)
    severity_results = severity_results[ ["path"] + severity_columns ] 

    covid_results = covid_results.join(severity_results.set_index('path'), on="path")
    
    non_covid_df = split_df[ split_df.has_covid == False]

    noncovid_results = app(
        pretrained=model_dir/"presence_f1.best.pkl",
        directory=Path(".."),
        scan=[Path(x) for x in non_covid_df.path],
        # scan_dir="../validation/non-covid/",
        output_csv=model_dir/"validation-noncovid.csv",
        output_mc=model_dir/"validation-noncovid.pt",
        # scan="../validation/non-covid/ct_scan_3",
        # scan_dir=[], # hack
        mc_samples=mc_samples,
        noncovid_txt=model_dir/"validation.tn.txt",
        covid_txt=model_dir/"validation.fp.txt",
        mc_dropout=mc_dropout,
    )
    noncovid_results["true"] = 0
    noncovid_results = noncovid_results.drop(columns=severity_columns)

    results = pd.concat([covid_results,noncovid_results])
    macro_f1 = f1_score(results["true"], results["COVID19 positive"], average="macro")
    f1_raw = f1_score(results["true"], results["COVID19 positive"], average=None)
    results.to_csv(model_dir/"validation.csv", index=False)
    print("macro_f1", macro_f1)
    print("f1_raw", f1_raw)
    # binary_f1 = f1_score(results["true"], results["COVID19 positive"])
    # print("binary_f1", binary_f1)

    severity_f1 = f1_score(severity_df["category"].str.lower(), severity_results["severity"].str.lower(), average="macro")

    print("severity_f1", severity_f1)

    with open(model_dir/"validation.macrof1.txt", 'w') as f:
        print('presence', macro_f1, file=f, sep=',')
        print('severity', severity_f1, file=f, sep=',')

    if log_wandb:
        with wandb.init(id=model_dir.name, project=app.project_name(), resume="must") as run:
            wandb.log({"presence_f1_final": macro_f1, "severity_f1_final": severity_f1})


if __name__ == "__main__":
    typer.run(main)
