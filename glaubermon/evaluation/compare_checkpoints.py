"""Compare frozen checkpoints on paired official games, reporting timeout contamination."""
import argparse
import json
from pathlib import Path
import numpy as np
from glaubermon.evaluation.summarize_benchmark import score


def compare(original, candidate):
    roots=[Path(original),Path(candidate)]
    manifests=[json.loads((p/'manifest.json').read_text()) for p in roots]
    for key in ('source','diff_sha256','games','seed','depth','max_turns','decision_seconds',
                'showdown_version','teams','actual_device','jobs','torch','hardware'):
        if manifests[0].get(key)!=manifests[1].get(key):
            raise ValueError('Comparison settings differ: '+key)
    results=[]
    for root,manifest in zip(roots,manifests):
        rows=[json.loads(f.read_text()) for f in root.glob('hybrid-[0-9][0-9][0-9].json')]
        rows=sorted(rows,key=lambda r:r['id'])
        if [r['id'] for r in rows]!=list(range(manifest['games'])):
            raise ValueError('Missing or duplicated hybrid games: '+str(root))
        results.append(rows)
    for left,right in zip(*results):
        for key in ('id','block','bot_side','seed','teams'):
            if left[key]!=right[key]:raise ValueError('Unpaired game: '+key)
    clean=all(r['status']=='completed' and not r['invalid_actions'] and not r['fallbacks'] for rows in results for r in rows)
    report=dict(purpose='Candidate minus original hybrid, paired by official game block',
                original_checkpoint=manifests[0]['checkpoint_sha256'],candidate_checkpoint=manifests[1]['checkpoint_sha256'],
                operationally_complete=clean,
                interpretation='Clock-limited score includes timeouts as losses; does not isolate playing strength' if not clean else 'All hybrid games completed; uncertainty applies only to these teams and seeds',
                settings={k:manifests[0].get(k) for k in ('games','depth','decision_seconds','actual_device','source')},
                runs=[],difference=None)
    for rows in results:
        report['runs'].append(dict(games=len(rows),completed=sum(r['status']=='completed' for r in rows),
                                  timeouts=sum(r['status']=='decision_timeout_forfeit' for r in rows),
                                  wins=sum(score(r)==1 for r in rows),unresolved=sum(score(r) is None for r in rows)))
    if all(score(r) is not None for rows in results for r in rows):
        blocks=sorted({r['block'] for r in results[0]})
        delta=np.array([np.mean([score(b)-score(a) for a,b in zip(*results) if a['block']==block]) for block in blocks])
        rng=np.random.default_rng(911)
        boot=rng.choice(delta,size=(20000,len(delta)),replace=True).mean(axis=1)
        report['difference']=dict(mean=float(delta.mean()),paired_block_95ci=np.quantile(boot,[.025,.975]).tolist())
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('original',type=Path);p.add_argument('candidate',type=Path);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError('Preserve previous report: select a new output')
    report=compare(a.original,a.candidate)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
