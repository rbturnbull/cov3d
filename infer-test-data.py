import typer
from pathlib import Path
from cov3d.apps import Cov3d

def main(model_dir:Path, mc_samples:int=0, mc_dropout:bool=False, presence_pretrained:str="presence_f1.best.pkl", severity_pretrained:str="severity_f1.best.pkl"):
    app = Cov3d()

    task1_dir = Path("../test-covid-ICASSP/task1/")
    tast1_scans = []
    for subset_index in range(1,2): # hack
    # for subset_index in range(1,12):
        subset_dir = task1_dir/f"subset{subset_index}"
        for path in subset_dir.iterdir():
            if path.is_dir():
                tast1_scans.append(path)

    predictions_dir = model_dir/"test-predictions"

    presence_results = app(
        pretrained=model_dir/presence_pretrained,
        # scan_dir="../validation/covid/",
        # scan=[Path("../validation/covid/ct_scan_156")],
        # scan_dir=[], # hack
        scan=tast1_scans,
        mc_samples=mc_samples,
        output_csv=predictions_dir/"presence-test-predictions.csv",
        output_mc=predictions_dir/"presence-test-predictions-mc.pt",
        noncovid_txt=predictions_dir/"non-covid.csv",
        covid_txt=predictions_dir/"covid.csv",
        mc_dropout=mc_dropout,
    )

    task2_dir = Path("../test-covid-ICASSP/task2/part_2_test_set_ICASSP_23/")
    tast2_scans = []
    for path in task2_dir.iterdir():
        if path.is_dir():
            tast2_scans.append(path)

    severity_results = app(
        pretrained=model_dir/severity_pretrained,
        scan=tast2_scans,
        mc_samples=mc_samples,
        output_csv=predictions_dir/"severity-test-predictions.csv",
        output_mc=predictions_dir/"severity-test-predictions-mc.pt",
        mild_txt=predictions_dir/"mild.csv",
        moderate_txt=predictions_dir/"moderate.csv",
        severe_txt=predictions_dir/"severe.csv",
        critical_txt=predictions_dir/"critical.csv",
        mc_dropout=mc_dropout,
    )    



if __name__ == "__main__":
    typer.run(main)
