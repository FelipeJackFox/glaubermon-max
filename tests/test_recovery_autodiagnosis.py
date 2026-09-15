import json
from pathlib import Path
import subprocess
import sys
import pytest
from glaubermon.evaluation import performance_recovery


@pytest.mark.parametrize('diagnosis_fails',[False,True])
def test_smoke_timeout_collects_profiles_without_starting_full_evaluation(tmp_path,monkeypatch,diagnosis_fails):
    original=tmp_path/'original.pt';candidate=tmp_path/'candidate.pt';trace=tmp_path/'trace.jsonl'
    for path in [original,candidate,trace]:path.write_text('fixture')
    showdown=tmp_path/'showdown';showdown.mkdir();(showdown/'package.json').write_text('{}')
    output=tmp_path/'out'; calls=[]
    def run(command,**kwargs):
        module=command[2].rsplit('.',1)[-1];calls.append((module,command))
        def arg(name):return command[command.index(name)+1]
        if module=='replay_decision':
            Path(arg('--output')).write_text(json.dumps({'seconds':1}))
        elif module=='official_benchmark':
            dest=Path(arg('--output'));dest.mkdir()
            (dest/'hybrid-000.json').write_text(json.dumps({'status':'decision_timeout_forfeit'}))
            assert arg('--games')=='2'
        elif module=='summarize_benchmark':
            (Path(command[3])/'summary.json').write_text(json.dumps({'modes':{'hybrid':{'completed_games':0,'invalid_actions':0,'fallbacks':0}}}))
        elif module=='diagnose_timeouts':
            assert arg('--checkpoint')==str(original)
            if diagnosis_fails:raise subprocess.CalledProcessError(2,command)
    monkeypatch.setattr(performance_recovery.subprocess,'run',run)
    monkeypatch.setattr(sys,'argv',['recovery','--original',str(original),'--candidate',str(candidate),
        '--original-trace',str(trace),'--candidate-trace',str(trace),'--showdown',str(showdown),
        '--output',str(output),'--devices','cpu'])
    with pytest.raises(SystemExit) as exc:performance_recovery.main()
    assert exc.value.code==2
    report=json.loads((output/'recovery.json').read_text())
    assert report['status']=='smoke_failed' and report['failed_checkpoint']=='original'
    assert (report['failure_diagnosis']=='completed') is (not diagnosis_fails)
    assert [name for name,_ in calls].count('official_benchmark')==1
    assert calls[-1][0]=='diagnose_timeouts'
