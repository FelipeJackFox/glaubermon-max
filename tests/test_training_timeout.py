"""Operational deadlines must not prune search or silently accept failed games."""
import json
from pathlib import Path

import pytest

from glaubermon.scripts import train_rebel
from glaubermon.evaluation import official_self_play


@pytest.mark.parametrize('seconds', [0, -1, float('nan'), float('inf'), -float('inf')])
def test_invalid_timeout_rejected_before_creating_experiment(tmp_path, seconds):
    output = tmp_path / 'invalid'
    with pytest.raises(ValueError, match='decision_seconds'):
        train_rebel.AlphaZeroTrainer(checkpoint_dir=str(output), decision_seconds=seconds)
    assert not output.exists()


def trainer(tmp_path, monkeypatch, seconds):
    monkeypatch.setattr(train_rebel.signal, 'signal', lambda *args: None)
    # No official process is needed for the forwarding/provenance checks.
    package = tmp_path / 'showdown'
    package.mkdir(exist_ok=True)
    (package / 'package.json').write_text('{"version":"0.11.11"}')
    return train_rebel.AlphaZeroTrainer(
        checkpoint_dir=str(tmp_path / 'run'), d_model=32, nhead=4,
        reset_from_scratch=True, showdown_path=str(package), decision_seconds=seconds)


def test_timeout_forwarded_recorded_and_compatible_on_resume(tmp_path, monkeypatch):
    captured = {}
    async def collect(*args, **kwargs):
        captured.update(kwargs)
        return [], {'terminated': True, 'turns': 1}
    monkeypatch.setattr(official_self_play, 'collect_game', collect)
    first = trainer(tmp_path, monkeypatch, 60)
    first.play_self_play_game()
    assert captured['decision_seconds'] == 60
    assert captured['depth'] == 1
    first.save_checkpoint()
    meta = json.loads(Path(first.meta_path).read_text())
    assert meta['decision_seconds'] == 60
    old_contract = first._data_contract()
    # Old metadata lacked the deadline: weights/counters remain compatible.
    meta.pop('decision_seconds')
    Path(first.meta_path).write_text(json.dumps(meta))
    resumed = train_rebel.AlphaZeroTrainer(
        checkpoint_dir=first.checkpoint_dir, d_model=32, nhead=4,
        showdown_path=first.showdown_path, decision_seconds=45)
    assert resumed._data_contract() == old_contract
    assert resumed.decision_seconds == 45


def test_timeout_aborts_without_training_or_counting_game(tmp_path, monkeypatch):
    async def fail(*args, **kwargs):
        Path(kwargs['trace_path']).write_text('{"frame":"failed decision"}\n')
        raise TimeoutError('test deadline')
    monkeypatch.setattr(official_self_play, 'collect_game', fail)
    instance = trainer(tmp_path, monkeypatch, 60)
    # Exercise the actual training loop while bypassing only the scope gate in this unit test.
    from glaubermon.evaluation import alignment_gate
    monkeypatch.setattr(alignment_gate, 'require_pilot_alignment', lambda *args: None)
    with pytest.raises(TimeoutError, match='game=0.*decision_seconds=60.*trace='):
        instance.train(games_to_play=1)
    assert instance.total_games == 0
    assert len(instance.replay_buffer) == 0
    assert not Path(instance.latest_ckpt).exists()
    assert (Path(instance.checkpoint_dir) / 'rollouts/game-000000.jsonl').exists()
