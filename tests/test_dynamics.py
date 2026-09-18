from dataclasses import replace
import math
import numpy as np
import pytest
from scipy.stats import kstest
from smo_sst.config import Config
from smo_sst.kernel import Simulator, library


def deterministic(**kwargs):
    return Config(noise_position=0., noise_heading=0., noise_speed_fraction=0.,
                  noise_adversary=0., p_fire=0., **kwargs)


def test_deterministic_unicycle_matches_scalar_equations_and_cost():
    c = deterministic(adversaries=0)
    sim = Simulator(c)
    s = sim.initial(np.empty((0,2)), 1)
    states = sim.rng_states(1, 1)
    x,y,theta,v,total = 0.,0.,0.,0.,0.
    for _ in range(17):
        dx,dy=20-x,30-y
        den=math.sqrt(dx*dx+dy*dy+c.smoothing**2)
        dx,dy=dx/den,dy/den
        den=math.sqrt(dx*dx+dy*dy+c.smoothing**2)
        dx,dy=dx/den,dy/den
        a=c.amax*math.tanh(2*(c.vmax-v)/c.vmax)
        w=c.omega_max*math.tanh(3*(-math.sin(theta)*dx+math.cos(theta)*dy))
        total+=c.dt*(1+.05*((a/c.amax)**2+(w/c.omega_max)**2))
        x,y=x+v*math.cos(theta)*c.dt,y+v*math.sin(theta)*c.dt
        theta,v=theta+w*c.dt,np.clip(v+a*c.dt,-c.vmax,c.vmax)
    out=sim.propagate(s,(20,30),17,states)
    np.testing.assert_allclose(out[0,:4],[x,y,theta,v],rtol=2e-6,atol=1e-6)
    assert out[0,-1] == pytest.approx(total, rel=2e-6)
    q,cost=sim.objective(out)
    assert q == 0
    assert cost == pytest.approx(total+c.duration*np.hypot(x-60,y-60)/(64*np.sqrt(2)), rel=2e-6)
    assert np.count_nonzero(s) == 0  # parent cloud was not mutated


def test_capture_absorbs_but_does_not_freeze_dynamics_or_cost():
    c=deterministic(adversaries=1,adv_speed=0.)
    sim=Simulator(c)
    s=sim.initial([[0,0]],2)
    s[:,7]=1
    out=sim.propagate(s,(10,10),1,sim.rng_states(7,2))
    assert np.all(out[:,-2]==1)
    after=sim.propagate(out,(10,10),10,sim.rng_states(8,2))
    assert np.all(after[:,-2]==1)
    assert np.all(after[:,-1]>out[:,-1])
    assert np.all(after[:,0]>out[:,0])
    assert sim.objective(after)[0]==1


def test_new_activation_and_termination_endpoint_semantics():
    c=replace(deterministic(adversaries=1,adv_speed=0.),p_fire=1.)
    sim=Simulator(c)
    s=sim.initial([[0,0]],1)
    launched=sim.propagate(s,(10,10),1,sim.rng_states(8,1))
    assert launched[0,7]==1 and launched[0,6]==0 and launched[0,-2]==1
    s[:,7]=1; s[:,6]=c.lifetime-1
    ended=sim.propagate(s,(10,10),1,sim.rng_states(8,1))
    assert ended[0,7]==2 and ended[0,6]==c.lifetime
    assert ended[0,-2]==0


def test_beta_sampler_distribution_is_bounded_non_gaussian_beta33():
    x=np.empty(80000)
    library().noise_samples(9841,x,len(x))
    assert np.all(np.abs(x)<1)
    assert abs(x.mean())<.005
    assert x.var()==pytest.approx(1/7,abs=.003)
    assert kstest((x+1)/2,"beta",args=(3,3)).pvalue>1e-4


def test_particle_results_independent_of_threads_and_edge_chunking():
    c=Config(adversaries=2,particles=64,rollout_threads=1)
    launchers=np.array([[1,1],[3,0]])
    a=Simulator(c)
    s=a.initial(launchers)
    states=a.rng_states(42,len(s))
    one=a.propagate(s,(10,10),20,states.copy())
    b=Simulator(replace(c,rollout_threads=4))
    segmented_states=states.copy()
    first=b.propagate(s,(10,10),7,segmented_states)
    two=b.propagate(first,(10,10),13,segmented_states)
    np.testing.assert_array_equal(one,two)
    assert np.all((two[:,3]>=-c.vmax)&(two[:,3]<=c.vmax))


def test_terminal_cost_replaces_previous_terminal_contribution():
    c=Config(adversaries=0)
    sim=Simulator(c)
    s=sim.initial(np.empty((0,2)),4)
    before=sim.objective(s)[1]
    out=sim.propagate(s,(60,60),10,sim.rng_states(1,4))
    terminal_before=before
    terminal_after=sim.objective(out)[1]-out[:,-1].mean(dtype=float)
    updated=before+out[:,-1].mean(dtype=float)+terminal_after-terminal_before
    assert updated==pytest.approx(sim.objective(out)[1])


@pytest.mark.parametrize("kwargs", [{"tau_max":0},{"confidence":1.},{"p_fire":-1.},
                                   {"particles":1.2},{"dt":float("nan")},{"rollout_threads":0}])
def test_config_rejects_invalid_values(kwargs):
    with pytest.raises(ValueError): Config(**kwargs)
