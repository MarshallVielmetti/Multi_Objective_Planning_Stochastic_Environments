from dataclasses import replace
from pathlib import Path
import json
import numpy as np
import pytest
from PIL import Image
from smo_sst.config import Config
from smo_sst.experiment import run_study,analyze


@pytest.fixture(scope="module")
def study(tmp_path_factory):
    out=tmp_path_factory.mktemp("study")/"run"
    c=Config(environments=2,iterations=30,particles=8,tau_max=3,adversaries=1,
             evaluation_particles=32,bootstrap_resamples=100,rollout_threads=1)
    run_study(c,out,save_tree=True)
    return out,c


def test_full_paired_pipeline_report_and_resume(study):
    out,c=study
    report=json.loads((out/"report.json").read_text())
    assert len(report["methods"])==4 and len(report["paired_differences"])==6
    before={p:p.stat().st_mtime_ns for p in out.glob("env-*/*.json")}
    run_study(c,out,resume=True)
    assert before=={p:p.stat().st_mtime_ns for p in out.glob("env-*/*.json")}
    for p in out.glob("env-*/*.json"):
        r=json.loads(p.read_text())
        assert 0<=r["evaluation_nhv"]<=1
        assert r["evaluation_nhv_interval"]["low"]<=r["evaluation_nhv"]<=r["evaluation_nhv_interval"]["high"]
        assert r["metrics"]["particle_steps"]>0
    assert (out/"table.tex").read_text().count("SMO-")==4
    with pytest.raises(ValueError,match="identical"):
        run_study(replace(c,seed=3),out,resume=True)


def test_partial_study_cannot_silently_drop_unfinished_environments(study,tmp_path):
    out,c=study
    (tmp_path/"manifest.json").write_text((out/"manifest.json").read_text())
    with pytest.raises(ValueError,match="incomplete"): analyze(tmp_path)


def test_publication_figures_and_animation_are_renderable(study):
    from smo_sst.visualize import figures,trajectory
    out,_=study
    paths=figures(out,dpi=65)
    paths+=trajectory(out,particles=3,dpi=45,fps=5,max_frames=4)
    assert all(Path(p).stat().st_size>100 for p in paths)
    im=Image.open(next(p for p in paths if p.endswith(".gif")))
    assert im.width>100 and im.height>100
