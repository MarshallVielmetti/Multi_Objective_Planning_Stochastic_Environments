import numpy as np
import pytest
from smo_sst.config import Config
from smo_sst.kernel import Simulator
from smo_sst.transport import signature, exact_distance


def test_transport_rejects_wrong_layout_before_entering_native_code():
    c = Config(adversaries=1)
    # A truncated cloud can still be reshaped as an adversary state, so checking
    # only that reshape is insufficient to protect the native ground-metric read.
    bad = np.zeros((4, c.state_dim-2), dtype=np.float32)
    with pytest.raises(ValueError, match="cloud"):
        signature(bad, c)
    with pytest.raises(ValueError, match="cloud"):
        exact_distance(bad, bad, c)


def test_replay_rejects_policies_outside_the_cost_bound_horizon():
    c = Config(adversaries=0, max_depth=2, tau_max=3)
    sim = Simulator(c)
    initial = np.empty((0, 2))
    for policy in ([{"waypoint": [1,1], "steps": 4}],
                   [{"waypoint": [1,1], "steps": 1}]*3,
                   [{"waypoint": [1,1], "steps": 1.5}]):
        with pytest.raises(ValueError, match="policy"):
            sim.replay(initial, policy, 2, 3)
