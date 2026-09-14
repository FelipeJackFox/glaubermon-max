"""Reconstruct one pending decision from a recorded player channel, without new battles.

Recorded choices restore the adapter's own action history. The opposite private
request is never delivered to the bot. cProfile is optional because it adds cost.
"""
import argparse
import asyncio
import cProfile
import hashlib
import json
import logging
from pathlib import Path
import platform
import pstats
import subprocess
import time

import torch
from glaubermon.client.showdown_bot import ShowdownBot
from glaubermon.models.set_transformer import GlaubermonMaxNet
from glaubermon.evaluation.showdown_transport import CapturedSocket


def inference_device(name='auto'):
    if name not in ('auto', 'cpu', 'cuda'):
        raise ValueError('device must be auto, cpu or cuda')
    if name == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable; refusing silent CPU fallback')
    return torch.device('cuda' if name == 'cuda' or (name == 'auto' and torch.cuda.is_available()) else 'cpu')


def restore_choice_history(bot, room, req, choice):
    if not req or req.get('wait') or req.get('teamPreview'):
        return
    if any(req.get('forceSwitch', [])):
        bot.last_action_was_switch[room] = False
        return
    if not choice:
        return
    tokens = choice.split()
    bot.last_action_was_switch[room] = tokens[0] == 'switch'
    moves = req.get('active', [{}])[0].get('moves', [])
    sucker = tokens[0] == 'move' and moves[int(tokens[1])-1]['id'] == 'suckerpunch'
    bot.sucker_punch_streak[room] = bot.sucker_punch_streak.get(room, 0)+1 if sucker else 0


async def replay(args):
    device = inference_device(args.device)
    torch.set_num_threads(1)
    model = GlaubermonMaxNet().to(device).eval()
    model.load_compatible_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True))
    bot = ShowdownBot(username='Glaubermon', depth=args.depth, model=model,
                      load_config=False, stealth=False, evaluator='hybrid')
    bot.ws = CapturedSocket()
    room = 'battle-replay'
    side = args.side-1
    req = None
    pending = False
    for line in Path(args.trace).read_text().splitlines():
        row = json.loads(line)
        if 'frame' in row:
            frame = row['frame']
            channel = frame['players'][side]
            bot.our_side[room] = f'p{args.side}'
            lines = [l for l in channel['lines'] if not l.startswith(('|request|','|win|','|tie|'))]
            await bot.handle_message('>'+room+'\n'+'\n'.join(lines))
            req = channel.get('request')
            pending = bool(req and not req.get('wait') and not req.get('teamPreview') and not frame['ended'])
            if pending:
                bot.build_battle_state(room, req)
        elif 'choices' in row:
            restore_choice_history(bot, room, req, row['choices'][side])
            pending = False
    if not pending:
        raise ValueError('Trace must end at an unanswered, non-preview decision for the selected side')

    original = bot.resolver.resolve_turn
    profiler = cProfile.Profile() if args.profile else None
    active = False
    solved = None
    forward_calls = 0
    def count(*unused):
        nonlocal forward_calls
        forward_calls += 1
    hook = model.register_forward_hook(count)
    def resolve(*a, **kw):
        nonlocal active, solved
        if active:
            return original(*a, **kw)
        active = True
        try:
            if profiler: profiler.enable()
            solved = original(*a, **kw)
            return solved
        finally:
            if profiler: profiler.disable()
            active = False
    bot.resolver.resolve_turn = resolve
    if device.type == 'cuda': torch.cuda.synchronize(device)
    started = time.perf_counter()
    try:
        await bot.handle_battle_turn(room, req)
        if device.type == 'cuda': torch.cuda.synchronize(device)
    finally:
        hook.remove()
    seconds = time.perf_counter()-started
    report = dict(trace_sha256=hashlib.sha256(Path(args.trace).read_bytes()).hexdigest(),
                  checkpoint_sha256=hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
                  source=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                  diff_sha256=hashlib.sha256(subprocess.check_output(['git','diff','HEAD'])).hexdigest(),
                  python=platform.python_version(),torch=torch.__version__,hardware=platform.platform(),
                  device=str(device),cuda_device=torch.cuda.get_device_name(device) if device.type=='cuda' else None,
                  turn=frame['turn'],side=args.side,depth=args.depth,force_switch=bool(req.get('forceSwitch')),
                  seconds=seconds,profile_enabled=bool(profiler),neural_forward_calls=forward_calls,
                  choices=bot.ws.messages,actions=[str(a) for a in solved[2]],
                  strategy=solved[1].tolist(),value=solved[3],purpose='One recorded pending decision; frozen checkpoint, no optimizer')
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError('Choose a new output path to preserve previous measurements')
    output.write_text(json.dumps(report,indent=2)+'\n')
    if profiler:
        profiler.dump_stats(str(output.with_suffix('.pstats')))
        with output.with_suffix('.profile.txt').open('w') as stream:
            pstats.Stats(profiler,stream=stream).sort_stats('cumtime').print_stats(40)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trace',type=Path,required=True)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--side',type=int,choices=[1,2],required=True)
    parser.add_argument('--depth',type=int,default=2)
    parser.add_argument('--device',choices=['auto','cpu','cuda'],default='cpu')
    parser.add_argument('--profile',action='store_true')
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    if args.depth<1:parser.error('depth must be positive')
    if args.output.exists():parser.error('output already exists; use a new file')
    logging.disable(logging.CRITICAL)
    print(json.dumps(asyncio.run(replay(args)),indent=2))


if __name__ == '__main__':main()
