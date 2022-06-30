import typer
from pathlib import Path
from sklearn.metrics import f1_score
import pandas as pd
from cov3d.apps import Covideo

def main(model_dir:Path, mc_samples:int=10):
    app = Covideo()
    covid_results = app(
        pretrained=model_dir/"export.pkl",
        scan_dir="../validation/covid/",
        # scan=[Path("../validation/covid/ct_scan_156")],
        # scan_dir=[], # hack
        scan=[],
        mc_samples=mc_samples,
        noncovid_txt=model_dir/"validation.fn.txt",
        covid_txt=model_dir/"validation.tp.txt",
    )
    covid_results["true"] = 1
    noncovid_results = app(
        pretrained=model_dir/"export.pkl",
        scan_dir="../validation/non-covid/",
        # scan="../validation/non-covid/ct_scan_3",
        # scan_dir=[], # hack
        scan=[],
        mc_samples=mc_samples,
        noncovid_txt=model_dir/"validation.tn.txt",
        covid_txt=model_dir/"validation.fp.txt",
    )
    noncovid_results["true"] = 0

    results = pd.concat([covid_results,noncovid_results])
    macro_f1 = f1_score(results["true"], results["COVID19 positive"], average="macro")
    results.to_csv(model_dir/"validation.csv", index=False)
    print("macro_f1", macro_f1)
    with open(model_dir/"validation.macrof1.txt", 'w') as f:
        print(macro_f1, file=f)
    # binary_f1 = f1_score(results["true"], results["COVID19 positive"])
    # print("binary_f1", binary_f1)


if __name__ == "__main__":
    typer.run(main)
