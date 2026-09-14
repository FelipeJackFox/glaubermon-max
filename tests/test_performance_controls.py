import re
from types import SimpleNamespace
import pytest
from glaubermon.core.constants import clean_key, _clean_key_text
from glaubermon.evaluation.replay_decision import inference_device, restore_choice_history


def test_cached_ids_preserve_normalization_and_bound_memory():
    for value in [None, '', 0, 42, [], ['Iron Treads'], 'Ogerpon-Wellspring', 'Électricité', 'Mr. Mime', 'THUNDER_CLAP', '♀']:
        expected=re.sub(r'[^a-zA-Z0-9]','',str(value)).lower() if value else ''
        assert clean_key(value)==expected
    for i in range(4200):clean_key(f'item-{i}')
    assert _clean_key_text.cache_info().currsize<=4096


def test_explicit_cpu_and_unavailable_cuda(monkeypatch):
    import torch
    monkeypatch.setattr(torch.cuda,'is_available',lambda:False)
    assert str(inference_device('cpu'))=='cpu'
    assert str(inference_device('auto'))=='cpu'
    with pytest.raises(RuntimeError,match='CUDA requested'):inference_device('cuda')


def test_replay_restores_own_choice_history():
    bot=SimpleNamespace(last_action_was_switch={},sucker_punch_streak={})
    req={'active':[{'moves':[{'id':'suckerpunch'},{'id':'ironhead'}]}]}
    restore_choice_history(bot,'room',req,'move 1')
    restore_choice_history(bot,'room',req,'move 1')
    assert bot.sucker_punch_streak['room']==2
    restore_choice_history(bot,'room',req,'switch 2')
    assert bot.last_action_was_switch['room'] and bot.sucker_punch_streak['room']==0
    restore_choice_history(bot,'room',{'forceSwitch':[True]},'switch 3')
    assert not bot.last_action_was_switch['room']
