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


def _cfg(tmp_path):
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
    return cfg


def test_identities(tmp_path):
    """Reconciled forecasts add up exactly; a run without every target ignores the identity."""
    import yaml
    for method in ("reconcile", "residual"):
        cfg = _cfg(tmp_path)
        cfg["identities"][0].update(method=method, residual="total_loans")
        cfg["methods"]["run"] = ["bma", "ar1"]
        p = tmp_path / "c.yaml"
        p.write_text(yaml.safe_dump(cfg))
        s = run.main(["--config", str(p), "--quick", "--no-report",
                      "--targets", "total_loans", "mortgage_balance", "commercial_balance"])
        m = "bma (identities)"
        tot, mo, co = (s[n]["scen"][m]["severe"] for n in ("total_loans", "mortgage_balance", "commercial_balance"))
        for b in tot:
            assert np.allclose(tot[b], mo[b] + co[b], rtol=1e-10)
        if method == "residual":     # the parts are untouched
            assert np.allclose(mo["Bank A"], s["mortgage_balance"]["scen"]["bma"]["severe"]["Bank A"])
    s = run.main(["--config", str(p), "--quick", "--no-report", "--targets", "mortgage_balance", "commercial_balance"])
    assert not any("(identities)" in m for m in s["mortgage_balance"]["lb"].index)
    assert s["mortgage_balance"]["identities"][0]["active"] is False


def test_new_methods(tmp_path):
    """Asymmetric BMA, monotone spline, hybrid and direct forecasts run and keep the expected signs."""
    import yaml
    cfg = _cfg(tmp_path)
    cfg["methods"]["run"] = ["bma", "bma_asym", "spline", "hybrid", "direct", "direct_class"]
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump(cfg))
    s = run.main(["--config", str(p), "--quick", "--no-report", "--targets", "loans_panel"])
    u = s["loans_panel"]
    assert {"bma_asym", "spline", "hybrid", "direct", "direct_class"} <= set(u["lb"].index)
    assert np.isfinite(u["lb"]["score"]).all()
    f, signs = u["fits"], u["signs"]
    for k, sl in f["spline"].info["slopes"].items():          # every piece keeps the expected sign
        sg = signs.get(k.split("@")[0], 0)
        assert all(x * sg >= -1e-12 for x in sl)
    for k, co in f["direct"].info["coef_h"].items():
        sg = signs.get(k.split("@")[0], 0)
        assert all(x * sg >= -1e-12 for x in co)
    assert len(f["direct"].info["persistence_h"]) == 12
    for m in ("spline", "hybrid", "direct"):
        assert all(np.isfinite(v).all() for v in u["scen"][m]["severe"].values())
