import numpy as np
import pytest
from smo_sst.metrics import normalized_hypervolume,mean_interval,pareto_indices


def test_hypervolume_known_rectangle_union_normalization_and_empty():
    p=[(.1,6),(.4,3),(.9,1),(.6,7),(.1,6)]
    assert normalized_hypervolume(p,10)==pytest.approx((.9*4+.6*3+.1*2)/10)
    assert normalized_hypervolume([],10)==0
    assert normalized_hypervolume([(1,0),(0,12)],10)==0
    assert normalized_hypervolume([(0,0)],10)==1
    assert normalized_hypervolume([(0,6)],10)==.4


def test_bootstrap_pairs_at_environment_level_and_handles_degeneracy():
    x=np.linspace(.2,.8,30)
    a=mean_interval(x,resamples=500,seed=1)
    assert a["low"]<.5<a["high"] and a["n"]==30
    b=mean_interval(x,resamples=500,seed=1)
    assert a==b
    paired=mean_interval((x+.1)-x,resamples=500,seed=1)
    assert paired["mean"]==pytest.approx(.1)
    assert paired["high"]-paired["low"]<1e-12
    assert mean_interval([.4])["low"] is None
    assert mean_interval([0,0])["low"]==mean_interval([0,0])["high"]==0


@pytest.mark.parametrize("p", [[(0,float("nan"))],[(-.1,1)],[(1.1,0)]])
def test_invalid_objectives_rejected(p):
    with pytest.raises(ValueError): normalized_hypervolume(p,10)
