import numpy as np
import pytest
from smo_sst.config import Config
from smo_sst.kernel import Simulator
from smo_sst.transport import signature,lower_bound,upper_bound,exact_distance,WitnessIndex


def random_cloud(rng,c,n):
    s=rng.normal(size=(n,c.state_dim)).astype(np.float32)
    for i in range(c.adversaries):
        s[:,6+4*i]=rng.integers(0,c.lifetime,n)
        s[:,7+4*i]=rng.integers(0,3,n)
    s[:,-2]=rng.integers(0,2,n)
    return s


def test_transport_bounds_bracket_exact_full_state_assignment():
    c=Config(adversaries=2)
    rng=np.random.default_rng(21)
    for _ in range(15):
        x,y=random_cloud(rng,c,16),random_cloud(rng,c,16)
        a,b=signature(x,c),signature(y,c)
        lo,exact,hi=lower_bound(a,b,c),exact_distance(x,y,c),upper_bound(a,b,c)
        assert lo<=exact+1e-12 and exact<=hi+1e-12
        if hi<=c.witness_radius:
            assert exact<=c.witness_radius+1e-12
        assert hi<=2+1e-12


def test_failure_is_not_a_zero_cost_sink_and_adversaries_are_labeled():
    c=Config(adversaries=2)
    sim=Simulator(c)
    a=sim.initial([[2,2],[4,4]],8)
    b=a.copy(); b[:,-2]=1
    assert exact_distance(a,b,c)==pytest.approx(1.)
    a[:,-2]=1; b[:,4]+=2
    assert exact_distance(a,b,c)>0
    b=a.copy(); b[:,7]=1
    assert exact_distance(a,b,c)==pytest.approx(c.mode_penalty/c.metric_diameter)
    b=a.copy(); b[:,4:8],b[:,8:12]=a[:,8:12],a[:,4:8]
    assert exact_distance(a,b,c)>0


def test_exact_distance_is_invariant_to_particle_permutation():
    c=Config(adversaries=2)
    rng=np.random.default_rng(8)
    a=random_cloud(rng,c,16)
    b=a[rng.permutation(len(a))].copy()
    assert exact_distance(a,b,c)==pytest.approx(0.)
    assert lower_bound(signature(a,c),signature(b,c),c)<1e-12


def test_embedding_index_matches_exhaustive_nearest_across_block_merges():
    rng=np.random.default_rng(4)
    index=WitnessIndex(block=8)
    points=[]
    for i in range(180):
        points.append(rng.normal(size=3)); index.add(points[-1])
        if i%11==0:
            query=rng.normal(size=3)
            want=np.argsort(np.linalg.norm(np.array(points)-query,axis=1))[:7]
            assert index.query(query,7)==want.tolist()
