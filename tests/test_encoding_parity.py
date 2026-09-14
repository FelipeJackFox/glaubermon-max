"""Byte-for-byte equivalence to the previous encoder over complete official trajectories."""
import importlib.util
from pathlib import Path
import pytest
import torch
from glaubermon.models.embeddings import encode_battle_state
from test_pilot_trajectories import CASES, run
from test_alignment_reference import from_snapshot

spec=importlib.util.spec_from_file_location('encoding_v7_reference',Path(__file__).parent/'fixtures/encoding_v7_reference.py')
legacy=importlib.util.module_from_spec(spec);spec.loader.exec_module(legacy)

@pytest.fixture(scope='module')
def trajectories():return {r['name']:r for r in run(CASES)}

@pytest.mark.parametrize('case',CASES,ids=lambda c:c['name'])
def test_all_features_identical_over_full_games(case,trajectories):
    row=trajectories[case['name']]
    for snapshot in [row['before']]+row['results']:
        state=from_snapshot(snapshot)
        new=encode_battle_state(state);old=legacy.encode_battle_state(state)
        for a,b in zip((*new[0],*new[1],new[2]),(*old[0],*old[1],old[2])):
            assert a.dtype==b.dtype==torch.float32
            assert torch.equal(a,b),(case['name'],snapshot['turn'])
        # Returned tensors must not alias a cache or another call.
        new[0][0].fill_(123)
        assert torch.equal(encode_battle_state(state)[0][0],old[0][0])
