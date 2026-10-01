"""
jev-net learn — hebbian credit assignment over the JEV transcript.

"Cells that fired together AND contributed load-bearing content, wire together
 stronger. Cells whose contributions the judge calls decorative, weaken."

CREDIT ASSIGNMENT (the scheme)
------------------------------
The judge (deepseek-v4-pro) receives the prompt + answer + the per-cell
contributions and returns, for EACH cell, a delta in {-1, 0, +1, +2}:
    +2 pivotal   — removing it breaks the answer
    +1 useful    — clearly touched the answer
     0 neutral   — decorative but harmless
    -1 noise     — off-target, wasted a hop
Additionally the judge returns an overall score 0-10 and up to 3 named gaps.

THE UPDATE, per edge A->B that actually delivered during the pass:
    signal = mean(salience_eff of delivered items) * B's delta / 2
    w     += eta * signal          (eta = 0.12)
    w     = clamp(w, 0.05, 1.0)
Edges into cells with delta +2 strengthen fastest; edges into noise cells
decay. Cells with NO incoming delivery this pass are untouched (no phantom
credit). n_updates increments on every touched edge — provenance of learning.

Thresholds also adapt, slightly: a cell that fired and earned +2 gets
threshold *= 0.97 (easier to fire next time); a noise cell gets *= 1.03.
Clamped to [0.05, 0.6]. This is the "neuron excitability" adaptation.

Everything is recorded as a learn receipt: {edge, before, after, delta_signal}
so learning itself is auditable — the ledger is the build.
"""
import json, time, os
from . import graph as G

DELTAS = {'pivotal': 2, 'useful': 1, 'neutral': 0, 'noise': -1}

JUDGE_SYS = """You are the judge of a JEV neural network pass. You see the original prompt,
the synthesized answer, and what each cell contributed. For EACH cell rate its contribution:
"pivotal" (+2, removing it breaks the answer) | "useful" (+1) | "neutral" (0, decorative) | "noise" (-1, off-target).
Also give overall score 0-10 and up to 3 gaps.
Reply ONLY JSON: {"cells": {"<cell>": "pivotal|useful|neutral|noise"}, "score": 0-10, "gaps": ["..."]}"""


def judge_contributions(prompt, answer, transcript, deadline_s=170):
    """deepseek-v4-pro judges each cell's contribution. Falls back to Qwen3.5-397B."""
    import sys as _sys
    _sys.path.insert(0, '/home/z/my-project/cot-quilt')
    try:
        from cot_decompose import http as cq_http, DS_KEY  # reuse the hardened wrapper
    except Exception:
        cq_http, DS_KEY = None, None
    contrib_lines = []
    for p in transcript:
        if p['cell'] in ('input', '__router__'): continue
        ex = '; '.join(f"({e['type']}) {e['content'][:140]}" for e in p['extractions'][:4])
        contrib_lines.append(f"[{p['cell']} @hop{p['hop']}]: verdict={p['verdict']['summary'][:160]} | parts: {ex[:420]}")
    usr = (f"PROMPT:\n{prompt[:1200]}\n\nANSWER:\n{(answer or '(no answer synthesized)')[:6000]}\n\n"
           f"CELL CONTRIBUTIONS:\n" + '\n'.join(contrib_lines)[:9000])
    body = {'model': 'deepseek-reasoner', 'messages': [{'role': 'system', 'content': JUDGE_SYS},
            {'role': 'user', 'content': usr}], 'max_tokens': 1400}
    if cq_http:
        r = cq_http('https://api.deepseek.com/chat/completions', DS_KEY, body, timeout=90)
        if r.get('status') == 200 and r['body'].get('choices'):
            ch = r['body']['choices'][0]['message']
            if ch.get('content'):
                return _parse_judge(ch['content'], 'deepseek-v4-pro'), r.get('ms')
    # fallback ladder via providers.chat
    from . import providers as PR
    r = PR.chat([{'role': 'system', 'content': JUDGE_SYS}, {'role': 'user', 'content': usr}],
                max_tokens=1400, deadline_s=deadline_s,
                ladder=[('Qwen3.5-397B', 'deepinfra', 'Qwen/Qwen3.5-397B-A17B'),
                        ('granite', 'deepinfra', 'ibm-granite/granite-4.2-3b')])
    if r['content']:
        return _parse_judge(r['content'], r['source']), r['ms']
    return {'cells': {}, 'score': None, 'gaps': ['judge unreachable']}, r['ms']


def _parse_judge(txt, source):
    try:
        d = json.loads(txt.strip().strip('`'))
        return {'cells': d.get('cells', {}), 'score': d.get('score'), 'gaps': d.get('gaps', []), 'judge': source}
    except Exception:
        s, e = txt.find('{'), txt.rfind('}')
        if s >= 0 and e > s:
            try:
                d = json.loads(txt[s:e+1])
                return {'cells': d.get('cells', {}), 'score': d.get('score'), 'gaps': d.get('gaps', []), 'judge': source}
            except Exception: pass
    return {'cells': {}, 'score': None, 'gaps': [f'judge output unparseable ({source})'], 'judge': source}


def _match_cell(judge_key, known_names):
    """Judges echo the labels we show them ('[builder @hop1]', 'echanism').
    Normalize + fuzzy-match back to the canonical cell name."""
    k = (judge_key or '').strip().strip('[]').strip()
    k = k.split('@')[0].strip().lower()
    for name in known_names:
        if k == name.lower(): return name
    for name in known_names:
        nl = name.lower()
        if nl in k or k in nl or nl.endswith(k) or k.endswith(nl):
            return name
    # subsequence match (handles truncation like 'echanism' -> 'mechanism')
    for name in known_names:
        nl = name.lower()
        it = iter(nl)
        if all(ch in it for ch in k):
            return name
    return None


def hebbian_update(net, transcript, judge, run_log, eta=0.12):
    """Apply the hebbian update from judge deltas to edge weights + thresholds.
    Returns the update receipt (list of per-edge before/after)."""
    known = [n for n in net['cells']]
    deltas, unmatched = {}, []
    for cell, label in (judge.get('cells') or {}).items():
        if label not in DELTAS: continue
        canon = _match_cell(cell, known)
        if canon is None:
            unmatched.append(cell); continue
        # a cell may fire at multiple hops; keep the MAX delta (best evidence)
        deltas[canon] = max(deltas.get(canon, -99), DELTAS[label])
    if unmatched:
        judge.setdefault('unmatched_keys', unmatched)
    updates = []
    for p in transcript:
        if p['cell'] in ('input', '__router__'): continue
        d_b = deltas.get(p['cell'], 0)
        # apportion credit across the edges that fed this cell, by weight share;
        in_e = G.in_edges(net, p['cell'])
        if not in_e: continue
        total_w = sum(x['weight'] for x in in_e) or 1.0
        for edge in in_e:
            # only credit edges whose accepts overlap the types actually delivered
            overlap = set(edge['accepts']) & set(p.get('notes', {}).get('delivered_types', []))
            if not overlap: continue
            share = edge['weight'] / total_w
            signal = share * (d_b / 2.0)
            before = edge['weight']
            edge['weight'] = round(max(0.05, min(1.0, before + eta * signal)), 4)
            edge['n_updates'] = edge.get('n_updates', 0) + 1
            updates.append({'edge': f"{edge['from']}->{edge['to']}", 'cell_delta': d_b,
                            'share': round(share, 3), 'signal': round(signal, 4),
                            'before': before, 'after': edge['weight']})
        # threshold adaptation
        c = net['cells'][p['cell']]
        if d_b >= 2: c['threshold'] = round(max(0.05, c['threshold'] * 0.97), 4)
        elif d_b <= -1: c['threshold'] = round(min(0.6, c['threshold'] * 1.03), 4)
    return {'updates': updates, 'judge': {k: v for k, v in judge.items() if k != 'cells'},
            'cell_deltas': deltas, 'eta': eta, 'ts': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}


def apply_and_save(net, learn_receipt, state_path, run_dir=None):
    if run_dir:
        os.makedirs(run_dir, exist_ok=True)
        tmp = os.path.join(run_dir, 'learn.json.tmp')
        with open(tmp, 'w') as f:
            json.dump(learn_receipt, f, indent=1); f.flush(); os.fsync(f.fileno())
        os.rename(tmp, os.path.join(run_dir, 'learn.json'))
    G.save_state(net, state_path)
