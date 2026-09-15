"""Profile CPU/CUDA, smoke-test a device, then evaluate both frozen checkpoints.

Never starts training, changes weights, or silently increases the decision limit.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import torch
from glaubermon.evaluation.compare_checkpoints import compare


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--candidate',type=Path,required=True)
    p.add_argument('--original',type=Path,default=Path('checkpoints/glaubermon_rebel_latest.pt'))
    p.add_argument('--candidate-trace',type=Path,required=True,help='Unanswered hybrid-000.jsonl from candidate evaluation')
    p.add_argument('--original-trace',type=Path,required=True,help='Unanswered hybrid-000.jsonl from original evaluation')
    p.add_argument('--showdown',type=Path,default=Path('tools/showdown/node_modules/pokemon-showdown'))
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--games',type=int,default=100,help='Games per mode after the two-game smoke test')
    p.add_argument('--devices',nargs='+',choices=['cpu','cuda'],default=None)
    p.add_argument('--decision-seconds',type=float,default=30)
    p.add_argument('--probe-seconds',type=float,default=180)
    a=p.parse_args()
    import math
    if a.games<2 or a.games%2:p.error('games must be positive and even, at least 2')
    if any(not math.isfinite(t) or t<=0 for t in (a.decision_seconds,a.probe_seconds)):p.error('timeouts must be finite and positive')
    for f in (a.original,a.candidate,a.original_trace,a.candidate_trace):
        if not f.is_file():p.error('Missing file: '+str(f))
    if not (a.showdown/'package.json').is_file():p.error('Install pinned Showdown first')
    devices=a.devices or (['cpu','cuda'] if torch.cuda.is_available() else ['cpu'])
    if 'cuda' in devices and not torch.cuda.is_available():p.error('CUDA explicitly requested but unavailable')
    a.output.mkdir(parents=True,exist_ok=False)
    report=dict(status='probing',devices=devices,probes={},failures=[],selected_device=None,
                hashes={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in (a.original,a.candidate,a.original_trace,a.candidate_trace)},
                decision_seconds=a.decision_seconds,requested_games=a.games)
    report_path=a.output/'recovery.json'
    def save():report_path.write_text(json.dumps(report,indent=2)+'\n')
    def run(module,arguments,name,timeout=None):
        cmd=[sys.executable,'-m','glaubermon.evaluation.'+module]+[str(v) for v in arguments]
        print('Running:', ' '.join(cmd),flush=True)
        with (a.output/(name+'.log')).open('w') as log:
            subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=timeout)
    save()
    checkpoints={'original':a.original,'candidate':a.candidate}
    traces={'original':a.original_trace,'candidate':a.candidate_trace}
    for device in devices:
        measured=[]
        for name in checkpoints:
            try:
                output=a.output/f'probe-{name}-{device}.json'
                run('replay_decision',['--trace',traces[name],'--checkpoint',checkpoints[name],
                                      '--side',1,'--depth',2,'--device',device,'--output',output],
                    f'probe-{name}-{device}',a.probe_seconds)
                measured.append(json.loads(output.read_text())['seconds'])
            except (subprocess.CalledProcessError,subprocess.TimeoutExpired) as exc:
                report['failures'].append(dict(stage='probe',device=device,checkpoint=name,error=str(exc)))
                break
        if len(measured)==2:report['probes'][device]=dict(seconds=measured,maximum=max(measured))
        save()
    if not report['probes']:
        report['status']='probe_failed';save();raise SystemExit(2)
    selected=min(report['probes'],key=lambda d:report['probes'][d]['maximum'])
    report['selected_device']=selected;report['status']='smoke';save()
    print('Selected device by worst recorded probe:',selected,flush=True)
    def benchmark(name,games,stage):
        dest=a.output/f'{stage}-{name}'
        run('official_benchmark',['--showdown',a.showdown,'--output',dest,'--checkpoint',checkpoints[name],
                                 '--games',games,'--jobs',1,'--depth',2,'--modes','hybrid','heuristic',
                                 '--seed',911,'--max-turns',300,'--decision-seconds',a.decision_seconds,'--device',selected],f'{stage}-{name}')
        run('summarize_benchmark',[dest],f'{stage}-{name}-summary')
        summary=json.loads((dest/'summary.json').read_text())
        return all(s['completed_games']==games and s['invalid_actions']==0 and s['fallbacks']==0 for s in summary['modes'].values())
    try:
        for name in checkpoints:
            if not benchmark(name,2,'smoke'):
                report.update(status='smoke_failed',failed_checkpoint=name);save()
                # The two-game smoke stage is bounded: collect the failure profile
                # in this same delivery, instead of requiring another operator run.
                failed_dir=a.output/f'smoke-{name}'
                failures=[json.loads(path.read_text()) for path in failed_dir.glob('hybrid-*.json')]
                if any(row.get('status')=='decision_timeout_forfeit' for row in failures):
                    try:
                        run('diagnose_timeouts',['--benchmark',failed_dir,'--checkpoint',checkpoints[name],
                            '--device',selected,'--output',a.output/f'diagnosis-{name}'],f'diagnosis-{name}')
                        report['failure_diagnosis']='completed'
                    except subprocess.CalledProcessError as exc:
                        report['failure_diagnosis']=str(exc)
                    save()
                raise SystemExit(2)
        if a.games>2:
            report['status']='evaluating';save()
            for name in checkpoints:
                if not benchmark(name,a.games,'evaluation'):
                    report.update(status='evaluation_incomplete',failed_checkpoint=name);save();raise SystemExit(2)
        stage='evaluation' if a.games>2 else 'smoke'
        result=compare(a.output/f'{stage}-original',a.output/f'{stage}-candidate')
        (a.output/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')
        report['status']='completed';save()
    except subprocess.CalledProcessError as exc:
        report.update(status='command_failed',error=str(exc));save();raise
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
