"""
jev-net forward — the inference pass.

    prompt -> INPUT cell (decompose into JEV-0)
           -> ROUTER projects via prism edges
           -> LAYER cells fire (LLM persona calls) -> emit JEV-1
           -> spawn verdicts may re-seed the router at hop+1  (JEV->JEV)
           -> SYNTH cell consumes everything that reached it
           -> transcript + receipts

Everything is receipted: every LLM attempt, every projection, every firing
decision, every spawn. The transcript is a DAG with parent links — the
hebbian learner reads exactly this structure.
"""
import json, time, os
from . import packets as P
from . import graph as G
from . import providers as PR


def _receipt_path(run_dir, name):
    return os.path.join(run_dir, name)


def atomic_write(path, obj):
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(obj, f, indent=1)
        f.flush(); os.fsync(f.fileno())
    os.rename(tmp, path)


def call_cell(net, cell, prompt_blocks, run_log, parents=(), hop=0, seed=None):
    """Fire one cell: build messages from persona + projected context, call the
    ladder, parse a JEV reply. Returns (packet|None, receipt)."""
    sys_p = (f"You are cell '{cell['name']}' inside a JEV neural network. {cell['persona']}\n"
             "Reply with ONLY a JSON object:\n"
             '{"judgment": {"verdict": "<what you conclude>", "confidence": 0.0-1.0, "grounds": "<why>"},\n'
             ' "extractions": [{"type": "sub-question|entity|constraint|mechanism|risk|claim|counter|anchor",\n'
             '                  "content": "<one granular component>", "salience": 0.0-1.0}],\n'
             ' "verdict": {"action": "route|spawn|answer|halt", "targets_hint": ["<cell names>"],\n'
             '             "summary": "<what to hand forward>"}}\n'
             'Emit 1-5 extractions. Confidence = how sure your verdict is. action "spawn" ONLY if you discovered a genuinely new component the net has not seen.')
    usr_p = '\n\n'.join(prompt_blocks)[:14000]
    r = PR.chat([{'role': 'system', 'content': sys_p}, {'role': 'user', 'content': usr_p}],
                max_tokens=cell.get('max_tokens', 900), deadline_s=cell.get('deadline_s', 60),
                ladder=cell.get('ladder'), seed_requested=seed)
    run_log['calls'].append({'cell': cell['name'], 'hop': hop, 'source': r['source'],
                             'ms': r['ms'], 'attempts': r['attempts'], 'seed_requested': seed})
    if not r['content']:
        return None, r
    try:
        parsed = json.loads(r['content'].strip().strip('`'))
    except Exception:
        # brace-balance harvest: truncated replies often only miss closing braces
        txt = r['content']
        s = txt.find('{')
        if s >= 0:
            cand = txt[s:]
            depth = 0; in_str = False; esc = False
            for ch in cand:
                if esc: esc = False; continue
                if ch == '\\': esc = True; continue
                if ch == '"': in_str = not in_str; continue
                if in_str: continue
                if ch == '{': depth += 1
                elif ch == '}': depth -= 1
            if depth > 0:
                cand = cand + '}' * depth
            try:
                parsed = json.loads(cand)
            except Exception:
                parsed = None
        else:
            parsed = None
    if parsed is None and r['attempts'] and r['attempts'][-1].get('finish') == 'length' and cell.get('max_tokens', 0) < 2400:
        # truncated JSON with more budget available: one repair retry, bigger budget
        bigger = dict(cell)
        bigger['max_tokens'] = min(2400, cell.get('max_tokens', 900) * 2)
        run_log['calls'].append({'cell': cell['name'], 'hop': hop, 'note': 'repair-retry (truncated)', 'source': r['source']})
        saved_mt, saved_dl = cell.get('max_tokens'), cell.get('deadline_s')
        cell['max_tokens'] = bigger['max_tokens']
        pkt2, r2 = call_cell(net, cell, prompt_blocks + ["Your previous reply was truncated. Return ONLY the complete JSON object, nothing else."], run_log, parents=parents, hop=hop, seed=seed)
        cell['max_tokens'] = saved_mt
        if pkt2:
            return pkt2, r2
    if parsed is None:
        return None, r
    pkt, repairs = P.packet_from_llm(parsed, cell['name'], hop, parents=parents)
    if pkt: pkt['notes']['repairs'] = repairs
    return pkt, r


def forward(net, prompt, run_dir=None, max_hops=2, spawn_budget=3, seed=None, typesafe_route=True):
    """One full forward pass. Returns the transcript dict (the receipt)."""
    run_id = f"fwd-{time.strftime('%H%M%S')}-{os.urandom(2).hex()}"
    run_dir = run_dir or f"runs/{run_id}"
    os.makedirs(run_dir, exist_ok=True)
    run_log = {'run_id': run_id, 'prompt': prompt, 'started': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
               'seed': seed, 'calls': [], 'hops': [], 'spawn_injections': [], 'packets': [],
               'max_hops': max_hops, 'spawn_budget': spawn_budget}
    transcript = []   # all packets, in firing order
    pending = []      # extraction dicts queued for the router

    # --- hop 0: INPUT cell decomposes the prompt --------------------------
    inp = net['cells'].get('input') or next(c for c in net['cells'].values() if c['role'] == 'input')
    pkt0, r = call_cell(net, inp,
        [f"PROMPT:\n{prompt}", "Decompose this prompt into granular components (sub-questions, entities, constraints, risks). Salience = how load-bearing each is for answering."],
        run_log, parents=[], hop=0, seed=seed)
    if pkt0:
        pkt0['cell'] = 'input'
        transcript.append(pkt0); run_log['packets'].append(pkt0['jev_id'])
        pending.extend(P.effective_salience(pkt0))
    else:
        # fail-open bootstrap: the prompt itself becomes one low-confidence extraction
        pending.append({'type': 'sub-question', 'content': prompt[:800], 'salience': 0.8, 'salience_eff': 0.8})
        pkt0 = None
    atomic_write(_receipt_path(run_dir, 'hop0.json'), {'input_packet': pkt0, 'pending': pending})

    spawns_used = 0
    for hop in range(1, max_hops + 1):
        # --- build synthetic router packet from pending extractions --------
        router_pkt = {'jev_id': P.new_id('route'), 'cell': '__router__', 'hop': hop,
                      'judgment': {'verdict': 'router projection', 'confidence': 1.0, 'grounds': 'deterministic'},
                      'extractions': pending, 'verdict': {'action': 'route', 'targets_hint': [], 'summary': ''},
                      'parents': [p['jev_id'] for p in transcript[-3:]], 'notes': {'synthetic': True}}
        inbox = G.router_project(net, pending)
        fires = G.who_fires(net, inbox)
        if not fires:
            fb = G.fallback_inject(net, inbox, router_pkt)
            if not fb:
                run_log['hops'].append({'hop': hop, 'note': 'dead hop, no eligible cells', 'fired': []})
                break
            fires = [fb]
        hop_rec = {'hop': hop, 'inbox': {k: {'activation': v['activation'], 'n_delivered': len(v['delivered'])} for k, v in inbox.items()},
                   'fired': [], 'spawned': []}
        pending = []  # consumed
        for name, slot in fires:
            cell = net['cells'][name]
            blocks = [f"ORIGINAL PROMPT:\n{prompt[:600]}"]
            # context = what the prism delivered, rendered
            ctx = P.bundle([{ 'jev_id': router_pkt['jev_id'], 'cell': '__router__', 'hop': hop,
                              'judgment': router_pkt['judgment'], 'extractions': slot['delivered'],
                              'verdict': router_pkt['verdict'], 'parents': [], 'notes': {}}], max_chars=4500)
            blocks.append(f"PROJECTED INTO YOU (activation {slot['activation']:.2f}):\n{ctx}")
            if hop > 1:
                prior = [p for p in transcript if p['hop'] == hop - 1][:4]
                if prior:
                    blocks.append("PRIOR LAYER VERDICTS:\n" + P.bundle(prior, max_chars=2600))
            pkt, r = call_cell(net, cell, blocks, run_log, parents=router_pkt['parents'], hop=hop, seed=seed)
            if not pkt:
                hop_rec['fired'].append({'cell': name, 'note': 'call failed', 'source': r.get('source')})
                continue
            pkt['notes']['activation_in'] = slot['activation']
            pkt['notes']['delivered_types'] = sorted({d['type'] for d in slot['delivered']})
            transcript.append(pkt); run_log['packets'].append(pkt['jev_id'])
            hop_rec['fired'].append({'cell': name, 'activation': slot['activation'],
                                     'action': pkt['verdict']['action'],
                                     'n_extractions': len(pkt['extractions'])})
            # ---- JEV->JEV trigger: spawn injects new components at hop+1 ----
            if pkt['verdict']['action'] == 'spawn' and spawns_used < spawn_budget and hop < max_hops:
                eff = P.effective_salience(pkt)
                new_items = [dict(e, salience_eff=min(1.0, e['salience_eff'] * 0.9)) for e in eff]
                pending.extend(new_items)
                spawns_used += 1
                hop_rec['spawned'].append({'from': name, 'n_new': len(new_items)})
                run_log['spawn_injections'].append({'hop': hop, 'from': name})
            # normal forward: non-spawn extractions also feed the next hop
            elif hop < max_hops:
                pending.extend(P.effective_salience(pkt))
        run_log['hops'].append(hop_rec)
        atomic_write(_receipt_path(run_dir, f'hop{hop}.json'), hop_rec)
        if not pending:
            break

    # --- SYNTH -------------------------------------------------------------
    synth = net['cells'].get('synth') or next(c for c in net['cells'].values() if c['role'] == 'synth')
    reached = [p for p in transcript if p['cell'] not in ('input', '__router__') and p['hop'] >= 1]
    blocks = [f"ORIGINAL PROMPT:\n{prompt}"]
    blocks.append("CELL CONTRIBUTIONS (JEV packets):\n" + (P.bundle(reached, 8000) if reached else "(none — answer alone)"))
    synth_sys = (f"You are cell '{synth['name']}', the synthesizer. Weave the cell contributions into ONE answer "
                 "to the original prompt. Cite which cell contributed which load-bearing part like: [skeptic] ... "
                 "Be concrete; if contributions conflict, surface the tension instead of averaging it away.")
    r = PR.chat([{'role': 'system', 'content': synth_sys}, {'role': 'user', 'content': '\n\n'.join(blocks)[:14000]}],
                max_tokens=synth.get('max_tokens', 1300), deadline_s=synth.get('deadline_s', 90),
                ladder=synth.get('ladder'), seed_requested=seed)
    run_log['calls'].append({'cell': 'synth', 'source': r['source'], 'ms': r['ms'], 'attempts': r['attempts']})
    answer = r['content']

    # --- typesafe route check (which entry framing best matches the answer) -
    route = None
    if typesafe_route and answer:
        route = PR.typesafe(state=(prompt + '\n\nANSWER:\n' + answer)[:16000],
                            questions=['Which single cell contribution was most load-bearing for this answer?',
                                       'What is the biggest remaining gap in this answer?'])
        run_log['calls'].append({'cell': 'typesafe-route', 'source': 'typesafe', 'ms': route.get('ms'),
                                 'status': route.get('status')})

    result = {'run_id': run_id, 'prompt': prompt, 'answer': answer, 'transcript': transcript,
              'route': route, 'run_log': run_log, 'finished': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
    atomic_write(_receipt_path(run_dir, 'result.json'), result)
    return result
