import json
import pytest
from glaubermon.evaluation.diagnose_timeouts import failed_decisions


def test_failed_side_is_restored_and_completed_games_excluded(tmp_path):
    (tmp_path/'manifest.json').write_text(json.dumps({'depth':2}))
    (tmp_path/'hybrid-000.json').write_text(json.dumps({'status':'completed'}))
    row={'status':'decision_timeout_forfeit','bot_side':1,'failed_decision':{'side':2,'turn':12}}
    (tmp_path/'hybrid-001.json').write_text(json.dumps(row))
    (tmp_path/'hybrid-001.jsonl').write_text('{}\n')
    _, cases=failed_decisions(tmp_path)
    assert len(cases)==1 and cases[0][1]['side']==2
    row['failed_decision']['side']=1
    (tmp_path/'hybrid-001.json').write_text(json.dumps(row))
    with pytest.raises(ValueError,match='Inconsistent player'):failed_decisions(tmp_path)


def test_missing_trace_does_not_silently_skip_failure(tmp_path):
    (tmp_path/'manifest.json').write_text('{}')
    (tmp_path/'hybrid-000.json').write_text(json.dumps({'status':'decision_timeout_forfeit'}))
    with pytest.raises(ValueError,match='Missing failed trace'):failed_decisions(tmp_path)
