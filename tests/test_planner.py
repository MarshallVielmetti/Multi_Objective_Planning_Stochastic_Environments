from dataclasses import replace
import numpy as np
from smo_sst.config import Config
from smo_sst.planner import Planner,Node,sparse_pareto_update


def node(i,q,c): return Node(i,1,1,q,c,None)


def test_strict_pareto_then_cell_cost_risk_creation_ties():
    a,b,c,d,e,f=(node(1,.11,10),node(2,.12,9),node(3,.13,11),
                 node(4,.12,9),node(5,.2,8),node(6,1.,2))
    assert [n.id for n in sparse_pareto_update([a,b,c,d,e,f],.1)]==[2,5,6]
    assert [n.id for n in sparse_pareto_update([d,b],.1)]==[2]


def test_inactive_ancestors_survive_until_last_child_removed():
    c=Config(adversaries=0)
    p=Planner(c,np.empty((0,2)),"reactive_sst",4)
    root=p.nodes[0]
    a=node(1,.1,4);a.parent=root;root.children=1
    b=node(2,.2,3);b.parent=a;a.children=1
    p.nodes.update({1:a,2:b})
    p.expandable += [a,b];p.locations.update({1:1,2:2})
    p._prune(a)
    assert 1 in p.nodes and 1 not in p.locations and not a.active
    assert len(b.policy())==2
    p._prune(b)
    assert list(p.nodes)==[0] and root.children==0
    assert p.expandable==[root]


def test_sequential_tree_invariants_fixed_centers_and_repeatability():
    c=Config(particles=16,adversaries=1,tau_max=5,max_depth=3,
             witness_radius=.5,iterations=160,rollout_threads=1)
    launchers=np.array([[2.,2.]])
    p=Planner(c,launchers,"reactive_sst",3)
    centers={}
    for _ in range(c.iterations):
        p.step()
        for depth,group in p.witnesses.items():
            for i,w in enumerate(group):
                key=(depth,i)
                if key in centers: np.testing.assert_array_equal(centers[key],w.center.cloud)
                else: centers[key]=w.center.cloud.copy()
                assert all(n.depth==depth and n.active for n in w.representatives)
                assert len(w.representatives)<=int(1/c.risk_resolution)+1
        assert p.nodes[0].active and p.nodes[0] in p.expandable
        assert all(n.active and n.depth<c.max_depth for n in p.expandable)
        assert all(n.parent is None or n.parent.id in p.nodes for n in p.nodes.values())
        assert all(n.children==sum(v.parent is n for v in p.nodes.values()) for n in p.nodes.values())
    other=Planner(c,launchers,"reactive_sst",3)
    for _ in range(c.iterations): other.step()
    assert [(n.id,n.risk,n.cost) for n in p.front()]==[(n.id,n.risk,n.cost) for n in other.front()]


def test_rrt_disables_only_pruning_and_depth_limit_keeps_root_expandable():
    c=Config(particles=8,adversaries=0,tau_max=3,max_depth=1,iterations=40)
    p=Planner(c,np.empty((0,2)),"target_rrt",3)
    r=p.run()
    assert r["nodes"]==41 and r["witnesses"]==0
    assert p.expandable==[p.nodes[0]]
    assert all(n.depth==1 for n in p.nodes.values() if n.id)


def test_exact_mode_uses_full_depth_witness_search():
    c=Config(particles=4,adversaries=0,tau_max=3,max_depth=2,iterations=15,distance="exact")
    r=Planner(c,np.empty((0,2)),"reactive_sst",2).run()
    assert r["metrics"]["omitted_witnesses"]==0
