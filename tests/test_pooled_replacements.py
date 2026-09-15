import numpy as np
import pytest
from test_volatile_phases_reference import references
from test_alignment_reference import from_snapshot
from glaubermon.core.actions import MoveAction
from glaubermon.search.subgame_resolver import SubgameResolver, simulate_turn_transition
from glaubermon.search.evaluators import HeuristicEvaluator


@pytest.mark.parametrize('case',[
    'faint_replacement_does_not_give_free_attack',
    'pivot_ko_chooses_before_foe_replacement',
    'replacement_hazard_ko_requires_another_choice',
    'pivot_resumes_queued_attack_on_incoming',
    'two_pivots_keep_both_replacement_choices',
])
def test_pooled_replacement_values_match_recursive_solver(case,references):
    state=from_snapshot(references[case]['before'])
    actions=[MoveAction(side.active_pokemon.moves[0].id,1) for side in (state.p1,state.p2)]
    phase=simulate_turn_transition(state,*actions)
    assert phase.pending_switches
    before=phase.clone()
    resolver=SubgameResolver(HeuristicEvaluator())
    expected=resolver._resolve_replacements(phase,0,False)[3]
    actual=resolver._evaluate_leaf_states([phase,state,phase])
    assert actual[0]==actual[2]==expected
    assert actual[1]==float(np.float32(resolver.evaluator.evaluate(state)))
    assert phase==before


def test_pool_limits_batch_memory_without_dropping_switch_options(references):
    state=from_snapshot(references['faint_replacement_does_not_give_free_attack']['before'])
    state.p2.pokemon.append(state.p2.pokemon[1].clone())
    state.p2.pokemon[-1].current_hp=50
    phase=simulate_turn_transition(state,MoveAction('seismictoss',1),MoveAction('splash',1))
    class Counting(HeuristicEvaluator):
        def __init__(self):self.sizes=[]
        def evaluate_batch(self,states):
            self.sizes.append(len(states))
            return super().evaluate_batch(states)
    evaluator=Counting();resolver=SubgameResolver(evaluator)
    expected=resolver._resolve_replacements(phase,0,False)[3]
    evaluator.sizes=[]
    values=resolver._evaluate_leaf_states([phase]*150)
    assert len(values)==150 and np.all(values==expected)
    assert sum(evaluator.sizes)==300 and max(evaluator.sizes)<=128
    assert len(evaluator.sizes)==3
