"""Fast end-to-end check: two targets, origins every 4th quarter. Run with: python -m pytest -q"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

import run  # noqa: E402


def test_quick_run(tmp_path):
    import yaml
    cfg = yaml.safe_load(open(os.path.join(HERE, "config.yaml")))
    cfg["output_dir"] = str(tmp_path)
    cfg["oos"]["start"] = "2020Q4"
    for k in ("metadata",):
        cfg["macro"][k] = os.path.join(HERE, cfg["macro"][k])
    cfg["macro"]["scenarios"] = {k: os.path.join(HERE, v) for k, v in cfg["macro"]["scenarios"].items()}
    for t in cfg["targets"]:
        for k in ("file", "events", "signs"):
            if t.get(k):
                t[k] = os.path.join(HERE, t[k])
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump(cfg))
    s = run.main(["--config", str(p), "--quick", "--targets", "loans_panel", "nii"])
    lb = s["loans_panel"]["lb"]
    assert {"bma", "combination", "class", "ar1", "drift", "random_walk"} <= set(lb.index)
    assert np.isfinite(lb["avg_err"]).all()
    assert "bma (rate x balance)" in s["nii"]["lb"].index
    pip = s["loans_panel"]["fits"]["bma"].info["pip"]
    assert max(pip.values()) <= 1 + 1e-9
    assert (tmp_path / "report.html").exists()


def test_options(tmp_path):
    """Bank by bank, chosen banks, restrictions, a tail metric and a custom benchmark."""
    import yaml
    cfg = yaml.safe_load(open(os.path.join(HERE, "config.yaml")))
    cfg["output_dir"] = str(tmp_path)
    cfg["oos"]["start"] = "2020Q4"
    cfg["macro"]["metadata"] = os.path.join(HERE, cfg["macro"]["metadata"])
    cfg["macro"]["scenarios"] = {k: os.path.join(HERE, v) for k, v in cfg["macro"]["scenarios"].items()}
    for t in cfg["targets"]:
        for k in ("file", "events", "signs"):
            if t.get(k):
                t[k] = os.path.join(HERE, t[k])
    cfg["estimation"] = {"mode": "separate", "banks": ["Bank A", "Bank C"]}
    cfg["restrictions"] = {"apply_to": ["bma", "combination", "class"], "force_in": ["FHPI.ION@1"],
                           "exclude": ["FHPI.IC"], "not_together": [["FYPD.ION", "FYPD.IC"]], "max_corr": None}
    cfg["ranking"] = {"metric": "cvar", "compare_against": ["class"]}
    cfg["methods"]["run"] = ["bma", "class"]
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump(cfg))
    s = run.main(["--config", str(p), "--quick", "--targets", "mortgage_balance", "--no-report"])
    assert set(s) == {"mortgage_balance | Bank A", "mortgage_balance | Bank C"}
    for u in s.values():
        lb = u["lb"]
        assert set(lb.index) == {"bma", "class"}
        assert (lb["score"] == lb["cvar"]).all() and lb["score"].is_monotonic_increasing
        assert "vs_class_pct" in lb.columns
        for e in u["fits"]["bma"].info["top"]:
            names = [v for v, _ in e["drivers"]]
            assert "FHPI.ION" in names and "FHPI.IC" not in names
            assert not ({"FYPD.ION", "FYPD.IC"} <= set(names))
