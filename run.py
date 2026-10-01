#!/usr/bin/env python3
"""jev-net CLI.

  python run.py forward "prompt..." [--net nets/default.json] [--hops 2] [--seed S]
  python run.py learn runs/fwd-XXXX [--net nets/default.json] [--state nets/state.json]
  python run.py show nets/state.json
"""
import sys, os, json, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from jevnet import graph as G, forward as F, learn as L, providers as PR


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['forward', 'learn', 'show', 'probe'])
    ap.add_argument('arg', nargs='?')
    ap.add_argument('--net', default=os.path.join(os.path.dirname(__file__), 'nets/default.json'))
    ap.add_argument('--state', default=os.path.join(os.path.dirname(__file__), 'nets/state.json'))
    ap.add_argument('--hops', type=int, default=2)
    ap.add_argument('--seed', type=int, default=None)
    ap.add_argument('--no-typesafe', action='store_true')
    a = ap.parse_args()

    if a.cmd == 'probe':
        ok, dt, d = None, None, None
        r = PR.moth_bits(4)
        print('moth:', r)
        return

    if a.cmd == 'show':
        print(json.dumps(G.load(a.arg or a.state), indent=1)[:3000])
        return

    if a.cmd == 'forward':
        net = G.load(a.net)
        seed = a.seed if a.seed is not None else PR.moth_seed()['seed']
        print(f'seed={seed} (16 quantum bits, honest bookkeeping)')
        res = F.forward(net, a.arg, max_hops=a.hops, seed=seed,
                        typesafe_route=not a.no_typesafe)
        print(json.dumps({'run_id': res['run_id'],
                          'answer': (res['answer'] or '')[:400],
                          'cells_fired': sorted({p['cell'] for p in res['transcript']}),
                          'route_status': (res['route'] or {}).get('status')}, indent=1))
        print('receipts ->', os.path.join('runs', res['run_id']))
        return

    if a.cmd == 'learn':
        run_dir = a.arg
        res = json.load(open(os.path.join(run_dir, 'result.json')))
        net = G.load(a.net)
        judge, jms = L.judge_contributions(res['prompt'], res['answer'], res['transcript'])
        print('judge:', json.dumps(judge)[:300], f'({jms}ms)')
        receipt = L.hebbian_update(net, res['transcript'], judge, res['run_log'])
        L.apply_and_save(net, receipt, a.state, run_dir)
        moved = [u for u in receipt['updates'] if abs(u['after'] - u['before']) > 1e-6]
        print(f'{len(moved)} edges updated, state -> {a.state}')
        for u in moved[:8]:
            print(f"  {u['edge']}: {u['before']} -> {u['after']} (delta {u['cell_delta']})")


if __name__ == '__main__':
    main()
