"""Summarize all scheduled games; bootstrap paired blocks, never individual games."""
import argparse
from collections import Counter
import json
from pathlib import Path
import numpy as np


def score(row):
    if row['status'] in ('truncated','phase_limit','infrastructure_error'):
        return None
    if row['winner'] == 'Glaubermon':
        return 1.0
    if row['winner'] == 'Control':
        return 0.0
    return 0.5 if row['status'] == 'completed' else None


def summarize(directory):
    directory=Path(directory)
    manifest=json.loads((directory/'manifest.json').read_text())
    rng=np.random.default_rng(911)
    summary=dict(source=manifest['source'], checkpoint_sha256=manifest['checkpoint_sha256'],
                 expected_per_mode=manifest['games'], modes={})
    block_scores={}
    for mode in manifest['modes']:
        rows=[json.loads(p.read_text()) for p in sorted(directory.glob(f'{mode}-[0-9][0-9][0-9].json'))]
        if {r['id'] for r in rows} != set(range(manifest['games'])):
            raise ValueError(f'{mode}: only {len(rows)}/{manifest["games"]} results; wait for every scheduled game')
        values=[score(r) for r in rows]
        latencies=[t for r in rows for t in r['latencies']]
        counts=Counter(r['status'] for r in rows)
        stats=dict(wins=values.count(1.0),losses=values.count(0.0),draws=values.count(0.5),
                   unresolved=values.count(None),statuses=dict(counts),
                   completed_games=counts['completed'], decision_timeouts=counts['decision_timeout_forfeit'],
                   latency_scope='Successful decisions only; timed-out decisions are recorded separately',
                   invalid_actions=sum(len(r['invalid_actions']) for r in rows),
                   fallbacks=sum(r['fallbacks'] for r in rows),
                   neural_forward_calls=sum(r.get('neural_forward_calls',0) for r in rows),
                   decisions=sum(r['decisions'] for r in rows),
                   latency_seconds={f'p{q}':float(np.percentile(latencies,q)) if latencies else None for q in (50,95,99,100)},
                   score_mean=None,block_bootstrap_95ci=None)
        if None not in values:
            blocks=np.array([np.mean([score(r) for r in rows if r['block']==b]) for b in sorted({r['block'] for r in rows})])
            bootstrap=rng.choice(blocks,size=(20000,len(blocks)),replace=True).mean(axis=1)
            stats['score_mean']=float(np.mean(values))
            stats['block_bootstrap_95ci']=np.quantile(bootstrap,[.025,.975]).tolist()
            block_scores[mode]=blocks
        summary['modes'][mode]=stats
    if set(block_scores) == {'hybrid','heuristic'}:
        differences=block_scores['hybrid']-block_scores['heuristic']
        boot=rng.choice(differences,size=(20000,len(differences)),replace=True).mean(axis=1)
        summary['hybrid_minus_heuristic']=dict(mean=float(differences.mean()),block_bootstrap_95ci=np.quantile(boot,[.025,.975]).tolist())
    (directory/'summary.json').write_text(json.dumps(summary,indent=2))
    lines=['# Piloto oficial Gen 9 OU','',
           f"Versión evaluada: `{manifest['source']}`; Showdown `{manifest['showdown_version']}`; poke-env `{manifest['poke_env']}`.",
           f"Dispositivo {manifest.get('actual_device','no registrado')}; límite {manifest['decision_seconds']} s. Profundidad {manifest['depth']}; {manifest['jobs']} procesos, PyTorch a 1 hilo por proceso. Checkpoint original respaldado; ningún peso entrenado con estas partidas.",
           '', '| Modo | Victorias | Derrotas | Empates | Sin resolver | Completadas | Timeouts | Puntuación | IC 95% por bloques |',
           '|---|---:|---:|---:|---:|---:|---:|---:|---|']
    for mode,stats in summary['modes'].items():
        point='—' if stats['score_mean'] is None else f"{100*stats['score_mean']:.1f}%"
        ci='—' if stats['block_bootstrap_95ci'] is None else '–'.join(f'{100*x:.1f}%' for x in stats['block_bootstrap_95ci'])
        lines.append(f"| {mode} | {stats['wins']} | {stats['losses']} | {stats['draws']} | {stats['unresolved']} | {stats['completed_games']} | {stats['decision_timeouts']} | {point} | {ci} |")
    if 'hybrid_minus_heuristic' in summary:
        diff=summary['hybrid_minus_heuristic']
        lo,hi=diff['block_bootstrap_95ci']
        lines += ['',f"Diferencia emparejada híbrido − heurísticas: {100*diff['mean']:.1f} puntos porcentuales; IC 95% {100*lo:.1f} a {100*hi:.1f}."]
    lines += ['', '## Integridad y latencia','']
    fmt = lambda x: '—' if x is None else f'{x:.3f}'
    for mode,stats in summary['modes'].items():
        lines += [f"- {mode}: {stats['decisions']} decisiones, {stats['invalid_actions']} registros de acciones inválidas, {stats['fallbacks']} elecciones default, {stats['neural_forward_calls']} llamadas reales a la red. Latencia p50 {fmt(stats['latency_seconds']['p50'])} s; p95 {fmt(stats['latency_seconds']['p95'])} s; máximo {fmt(stats['latency_seconds']['p100'])} s."]
    lines += ['', '## Alcance','',
              f"Son {manifest['games']//2} bloques de dos partidas por modo. Se intercambian agentes entre lados y equipos con la misma semilla. El bootstrap remuestrea bloques completos, con 20,000 réplicas.",
              '', 'Solo se usaron los tres equipos entregados que pasan la validación OU de esta versión: balance, stall y pelol. Hyper offense fue excluido antes de congelar el piloto porque Roaring Moon está prohibido. Los equipos, semillas, configuración, hashes y política de fallos están en manifest.json. Cada partida conserva resultado, latencias, elecciones, canales privados separados y log público.',
              '', 'La incertidumbre describe este conjunto reducido de equipos y semillas. No mide toda la ladder ni certifica SOTA. El historial de entrenamiento del checkpoint original puede incluir estos equipos; no se afirma que fueran desconocidos para ese modelo. Estas partidas quedan reservadas para evaluación y no alimentan el nuevo entrenador.',
              '', 'Un resultado sin terminar conserva estado sin resolver; nunca se convierte en empate. Los fallos de infraestructura no se ocultan. Las derrotas por timeout sí cuentan en la puntuación con reloj: no permiten aislar fuerza de juego sin límite. Los percentiles de latencia excluyen decisiones abortadas. El checkpoint evaluado y el dispositivo se identifican en manifest.json.']
    (directory/'RESULTADOS.md').write_text('\n'.join(lines)+'\n')
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory')
    print(json.dumps(summarize(parser.parse_args().directory),indent=2))
