from dataclasses import replace
import json
from pathlib import Path
import numpy as np
import pytest
from PIL import Image
from smo_sst.config import Config, BASELINE_METHODS, METHODS, POLICY_FAMILIES
from smo_sst.kernel import Simulator
from smo_sst.planner import Planner
from smo_sst.policy import sample_gains, survival_table
from smo_sst.experiment import run_study, evaluate, seed_for


def encounter(age=0., behind=False):
    c = Config(adversaries=1, noise_position=0., noise_heading=0., noise_speed_fraction=0.,
               noise_adversary=0., p_fire=0.)
    sim = Simulator(c, family="predictive")
    cloud = sim.initial([[-4 if behind else 4, 0]], 1)
    cloud[0, 3] = c.vmax
    cloud[0, 6:8] = [age, 1]
    return c, sim, cloud


def test_tangential_policy_breaks_head_on_symmetry_with_both_passing_sides():
    c, sim, cloud = encounter()
    radial = sim.controls(cloud, (60,0), (2.,0.))[0]
    left = sim.controls(cloud, (60,0), (2.,-2.))[0]
    right = sim.controls(cloud, (60,0), (2.,2.))[0]
    assert radial[1] == pytest.approx(0.)
    assert left[1] > .1 and right[1] < -.1
    assert left[1] == pytest.approx(-right[1])
    assert left[0] == radial[0] == right[0] == 0.  # speed law is unchanged
    out = sim.propagate(cloud, (60,0), 1, sim.rng_states(1,1), (2.,2.))
    assert out[0,2] == pytest.approx(right[1]*c.dt, abs=1e-8)


def test_urgency_accounts_for_interception_geometry_and_remaining_lifetime():
    c, sim, head_on = encounter()
    _, _, behind = encounter(behind=True)
    _, _, expiring = encounter(age=c.lifetime-1)
    approaching_turn = abs(sim.controls(head_on, (60,0), (1.,2.))[0,1])
    receding_turn = abs(sim.controls(behind, (60,0), (1.,2.))[0,1])
    expiring_turn = abs(sim.controls(expiring, (60,0), (1.,2.))[0,1])
    assert approaching_turn > 5*receding_turn
    assert expiring_turn == 0.  # certainly terminal before the predicted encounter
    assert approaching_turn > .1


def test_survival_matches_products_of_the_actual_transition_hazards():
    L = 54
    table = survival_table(L)
    assert not table.flags.writeable
    assert np.all(table[:,0] == 1) and np.all(table[:,-1] == 0)
    assert table[0,40] == 1
    for age, steps in [(0,47),(43,3),(49,2),(52,1),(53,1)]:
        hazards = [np.clip((age+j-.8*(L-1))/(.2*(L-1)), 0, 1) for j in range(steps)]
        assert table[age,steps] == pytest.approx(np.prod(1-np.array(hazards)))
    assert np.all(np.diff(table,axis=1) <= 0)


def test_control_is_continuous_at_age_knots_and_bounded_at_zero_separation():
    c, sim, cloud = encounter(age=42)
    cloud[0,4] = .5
    below, above = cloud.copy(), cloud.copy()
    below[0,6] -= 1e-4; above[0,6] += 1e-4
    u1 = sim.controls(below, (60,0), (2.,2.))
    u2 = sim.controls(above, (60,0), (2.,2.))
    np.testing.assert_allclose(u1,u2,atol=1e-4)
    cloud[0,4:6] = 0
    u = sim.controls(cloud, (0,0), (2.,2.))
    assert np.isfinite(u).all()
    assert abs(u[0,0]) <= c.amax and abs(u[0,1]) <= c.omega_max


def test_zero_gains_reproduce_target_rollouts_bitwise_with_stochastic_adversaries():
    c = Config(adversaries=2, particles=64)
    target = Simulator(c, family="target")
    pred = Simulator(c, family="predictive")
    cloud = target.initial([[1,1],[2,0]])
    rng = target.rng_states(12,len(cloud))
    a = target.propagate(cloud,(30,30),30,rng.copy())
    b = pred.propagate(cloud,(30,30),30,rng.copy(),(0.,0.))
    np.testing.assert_array_equal(a,b)


def test_gain_sampling_includes_baseline_atom_and_both_tangential_directions():
    c = Config()
    rng = np.random.default_rng(6)
    samples = np.array([sample_gains(rng,c) for _ in range(10000)])
    zero = np.all(samples==0,axis=1)
    assert zero.mean() == pytest.approx(.2,abs=.015)
    assert np.all((samples[:,0]>=0)&(samples[:,0]<=2))
    assert np.all(np.abs(samples[:,1])<=2)
    assert (samples[:,1]>1).any() and (samples[:,1]<-1).any()


def test_baseline_atom_only_keeps_parent_duration_noise_and_pruning_streams_identical():
    c = Config(adversaries=2, particles=16, tau_max=10, iterations=90,
               predictive_baseline_probability=1.)
    target = Planner(c,[[1,1],[3,0]],"target_sst",42)
    pred = Planner(c,[[1,1],[3,0]],"predictive_sst",42)
    for _ in range(c.iterations):
        a, accept_a = target.step()
        b, accept_b = pred.step()
        assert (a.parent.id,a.waypoint,a.steps,a.risk,a.cost,accept_a) == (
                b.parent.id,b.waypoint,b.steps,b.risk,b.cost,accept_b)
        assert b.gains == (0.,0.)
        np.testing.assert_array_equal(a.cloud,b.cloud)
    assert set(target.nodes) == set(pred.nodes)


def test_saved_predictive_gains_roundtrip_and_replay_are_thread_and_chunk_invariant():
    c = Config(adversaries=1, max_depth=3, tau_max=20, rollout_threads=1)
    policy = [{"waypoint":[30,30],"steps":17,"radial_gain":1.2,"tangential_gain":-1.8},
              {"waypoint":[10,30],"steps":11,"radial_gain":.4,"tangential_gain":.8}]
    one = Simulator(c,family="predictive").replay([[1,1]],policy,64,9)
    restored = json.loads(json.dumps(policy))
    two, _ = Simulator(replace(c,rollout_threads=4),family="predictive").replay(
        [[1,1]],restored,64,9,trace=True)
    np.testing.assert_array_equal(one,two)
    target = Simulator(c,family="target").replay([[1,1]],
        [{"waypoint":e["waypoint"],"steps":e["steps"]} for e in policy],64,9)
    assert not np.array_equal(one,target)


def test_predictive_evaluation_uses_the_recorded_controller_and_gains():
    c = Config(adversaries=1,evaluation_particles=64)
    policy = [{"waypoint":[30,30],"steps":40,"radial_gain":1.,"tangential_gain":2.}]
    run = {"method":"predictive_sst","front":[{"node":1,"policy":policy}]}
    evaluate(run,c,[[1,1]],12)
    sim = Simulator(c,family="predictive")
    cloud = sim.replay([[1,1]],policy,c.evaluation_particles,seed_for(12,0,71))
    q,cost = sim.objective(cloud)
    assert run["evaluation"][0]["risk"] == q
    assert run["evaluation"][0]["cost"] == cost


def test_new_group_runs_reports_and_animates_without_changing_default_baselines(tmp_path):
    from smo_sst.visualize import figures, trajectory
    assert len(BASELINE_METHODS)==4 and METHODS[-1]=="predictive_sst"
    assert POLICY_FAMILIES["predictive_sst"]=="predictive"
    c = Config(environments=2,iterations=40,particles=8,tau_max=100,adversaries=1,
               evaluation_particles=32,bootstrap_resamples=100)
    out = tmp_path/"study"
    report = run_study(c,out,methods=("target_sst","predictive_sst"))
    assert "target_sst - predictive_sst" in report["paired_differences"]
    raw = json.loads((out/"env-000/predictive_sst.json").read_text())
    edges = [e for v in raw["front"] for e in v["policy"]]
    assert edges and all("radial_gain" in e and "tangential_gain" in e for e in edges)
    assert "Predictive SMO-SST" in (out/"report.md").read_text()
    paths = figures(out,dpi=40)
    paths += trajectory(out,method="predictive_sst",particles=2,dpi=30,fps=5,max_frames=3)
    assert all(Path(p).stat().st_size>100 for p in paths)
    assert Image.open(next(p for p in paths if p.endswith(".gif"))).width>100
    assert json.loads((out/"manifest.json").read_text())["policy_families"]["predictive_sst"]=="predictive"


@pytest.mark.parametrize("bad", [None,(1.,),(float("nan"),1.),(-1.,1.),(1.,3.)])
def test_invalid_or_missing_predictive_gains_are_rejected(bad):
    _,sim,cloud = encounter()
    with pytest.raises(ValueError,match="gain"):
        sim.controls(cloud,(10,10),bad)


def test_ambiguous_saved_policy_cannot_silently_replay_as_target_only():
    c = Config(adversaries=1)
    sim = Simulator(c,family="predictive")
    with pytest.raises(ValueError,match="record"):
        sim.replay([[1,1]],[{"waypoint":[20,20],"steps":4}],2,1)
    with pytest.raises(ValueError,match="baseline"):
        Simulator(c,family="target").replay([[1,1]],
            [{"waypoint":[20,20],"steps":4,"radial_gain":1.,"tangential_gain":1.}],2,1)


@pytest.mark.parametrize("kwargs", [{"predictive_horizon":0.},{"predictive_guard_radius":.1},
    {"predictive_velocity_epsilon":0.},{"predictive_radial_max":-1.},
    {"predictive_tangential_max":float("nan")},{"predictive_baseline_probability":1.1}])
def test_predictive_configuration_validation(kwargs):
    with pytest.raises(ValueError): Config(**kwargs)
