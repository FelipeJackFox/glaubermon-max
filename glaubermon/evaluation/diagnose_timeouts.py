"""Replay every timed-out decision of one benchmark, preserving failed evidence.

Run unprofiled first, then with cProfile. Never starts games or training.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys


def failed_decisions(directory):
    manifest = json.loads((directory / 'manifest.json').read_text())
    cases = []
    for path in sorted(directory.glob('hybrid-*.json')):
        row = json.loads(path.read_text())
        if row.get('status') != 'decision_timeout_forfeit':
            continue
        trace = path.with_suffix('.jsonl')
        if not trace.is_file():
            raise ValueError(f'Missing failed trace: {trace}')
        failure = row['failed_decision']
        if failure['side'] != row['bot_side'] + 1:
            raise ValueError(f'Inconsistent player side: {path}')
        cases.append((trace, failure))
    if not cases:
        raise ValueError('No recorded hybrid timeouts in this directory')
    return manifest, cases


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--benchmark', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--device', choices=['cpu', 'cuda'], required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--process-seconds', type=float, default=300)
    a = p.parse_args()
    if not math.isfinite(a.process_seconds) or a.process_seconds <= 0:
        p.error('process-seconds must be finite and positive')
    manifest, cases = failed_decisions(a.benchmark)
    digest = hashlib.sha256(a.checkpoint.read_bytes()).hexdigest()
    if digest != manifest['checkpoint_sha256']:
        p.error('Checkpoint does not match the failed benchmark')
    a.output.mkdir(parents=True, exist_ok=False)
    report = dict(status='running', checkpoint_sha256=digest, benchmark_source=manifest['source'],
                  device=a.device, decision_seconds=manifest['decision_seconds'], cases=[],
                  purpose='Diagnostic only; process timeout is not a game decision allowance')
    def save():
        (a.output / 'diagnosis.json').write_text(json.dumps(report, indent=2)+'\n')
    save()
    failed = False
    for trace, failure in cases:
        case = dict(trace_sha256=hashlib.sha256(trace.read_bytes()).hexdigest(),
                    original_failure=failure, measurements=[])
        report['cases'].append(case)
        for profile in (False, True):
            name = trace.stem + ('-profile' if profile else '-timing')
            output = a.output / (name+'.json')
            command = [sys.executable, '-m', 'glaubermon.evaluation.replay_decision',
                       '--trace', str(trace), '--checkpoint', str(a.checkpoint),
                       '--side', str(failure['side']), '--depth', str(manifest['depth']),
                       '--device', a.device, '--output', str(output)]
            if profile:
                command.append('--profile')
            print('Running:', name, flush=True)
            item = dict(profile=profile, output=output.name)
            try:
                with (a.output / (name+'.log')).open('w') as log:
                    subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                   check=True, timeout=a.process_seconds)
                result = json.loads(output.read_text())
                item.update(status='completed', seconds=result['seconds'],
                            neural_forward_calls=result['neural_forward_calls'])
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
                item.update(status='failed', error=str(exc))
                failed = True
            case['measurements'].append(item)
            save()
    report['status'] = 'incomplete' if failed else 'completed'
    save()
    if failed:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
