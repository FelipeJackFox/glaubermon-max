import json
import pytest
from glaubermon.evaluation.compare_checkpoints import compare
from glaubermon.evaluation.summarize_benchmark import summarize


def make_run(root,timeout=False):
    root.mkdir()
    (root/'manifest.json').write_text(json.dumps(dict(source='test',checkpoint_sha256=root.name,games=2,modes=['hybrid'],showdown_version='0.11.11',poke_env='test',depth=2,jobs=1,decision_seconds=30)))
    for i in range(2):
        row=dict(id=i,block=0,bot_side=i,seed=[1,2,3,4],teams=['balance','stall'],status='decision_timeout_forfeit' if timeout else 'completed',winner='Control' if timeout else 'Glaubermon',invalid_actions=[],fallbacks=0,decisions=0,latencies=[])
        (root/f'hybrid-{i:03d}.json').write_text(json.dumps(row))


def test_empty_latency_and_timeout_are_explicit(tmp_path):
    p=tmp_path/'timeouts';make_run(p,True)
    s=summarize(p)['modes']['hybrid']
    assert s['decision_timeouts']==2 and s['completed_games']==0
    assert s['latency_seconds']['p95'] is None and s['losses']==2
    assert '1 bloques' in (p/'RESULTADOS.md').read_text()


def test_checkpoint_comparison_reports_clock_losses_and_rejects_drift(tmp_path):
    a=tmp_path/'old';b=tmp_path/'new';make_run(a);make_run(b,True)
    result=compare(a,b)
    assert not result['operationally_complete'] and result['difference']['mean']==-1
    manifest=json.loads((b/'manifest.json').read_text());manifest['decision_seconds']=60
    (b/'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='decision_seconds'):compare(a,b)
