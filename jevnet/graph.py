"""
jev-net graph — cells, edges, and the prism projection.

TOPOLOGY SEMANTICS
------------------
A net is {cells: {name -> cell}, edges: [{from, to, accepts, weight}]}.

  cell = {
    'name':     'skeptic',
    'role':     'layer',        # 'input' | 'layer' | 'router' | 'synth' | 'judge'
    'persona':  '...',          # the system prompt spine for the LLM call
    'accepts':  ['counter','risk','sub-question'],   # extraction types it drinks
    'threshold': 0.12,           # fires iff incoming activation >= threshold
    'ladder':   [['seed-mini','deepinfra','ByteDance/Seed-2.0-mini'], ...],
    'max_tokens': 900, 'deadline_s': 60
  }

  edge = {'from': 'input', 'to': 'skeptic', 'accepts': ['sub-question','constraint'],
          'weight': 0.8}      # weight in [0.05,1.0], hebbian-learned

PRISM PROJECTION (the forward edge math)
----------------------------------------
When cell A emits packet P, for each edge A->B:

    delivered(P, A->B) = [e for e in P.extractions
                          if e.type in edge.accepts and e.type in B.accepts]
    activation(B)     += sum(e.salience * A.confidence * edge.weight
                             for e in delivered)

That triple product — salience x confidence x weight — is the whole learning
surface: salience is the prism separating wavelengths, confidence is the
presynaptic firing strength, weight is the synaptic strength we later update.

A cell B fires iff activation(B) >= B.threshold AND it received at least one
delivered extraction. If a hop produces ZERO firing cells, the pass falls
through: the router re-injects the highest-salience undelivered extraction
into the strongest-weighted eligible cell (guaranteed progress, receipted).

JEV->JEV ITERATION
------------------
verdict.action == 'spawn' packets inject their NEW extractions into the router
at hop+1, subject to hop <= max_hops and a spawn budget per pass. This is the
iterator semantics — the net can grow its own middle layers mid-answer.
"""
import json, os

TYPES = ('sub-question', 'entity', 'constraint', 'mechanism', 'risk', 'claim', 'counter', 'anchor')


def load(path):
    net = json.load(open(path))
    return normalize(net)


def normalize(net):
    for name, c in net['cells'].items():
        c.setdefault('name', name)  # key becomes the identity
        c.setdefault('threshold', 0.12)
        c.setdefault('role', 'layer')
        c.setdefault('accepts', list(TYPES))
        c.setdefault('max_tokens', 900)
        c.setdefault('deadline_s', 60)
    for e in net['edges']:
        e['weight'] = max(0.05, min(1.0, float(e.get('weight', 0.7))))
        e.setdefault('accepts', list(TYPES))
        e.setdefault('n_updates', 0)
    return net


def out_edges(net, cell):
    return [e for e in net['edges'] if e['from'] == cell]


def in_edges(net, cell):
    return [e for e in net['edges'] if e['to'] == cell]


def project(net, packet):
    """Prism projection of one packet onto downstream cells.
    Returns {cell_name: {'activation': float, 'delivered': [extractions...],
                         'via': [edge refs...]}} — only cells with >0 delivered."""
    inbox = {}
    for e in effective(packet):
        for edge in out_edges(net, packet['cell']):
            target = net['cells'].get(edge['to'])
            if not target: continue
            if e['type'] not in edge['accepts'] or e['type'] not in target['accepts']:
                continue  # the prism: this wavelength does not pass
            a = e['salience'] * packet['judgment']['confidence'] * edge['weight']
            slot = inbox.setdefault(edge['to'], {'activation': 0.0, 'delivered': [], 'via': []})
            slot['activation'] += a
            slot['delivered'].append({**e, 'via_edge': f"{edge['from']}->{edge['to']}", 'w': edge['weight']})
            slot['via'].append(f"{edge['from']}->{edge['to']}")
    return {k: {'activation': round(v['activation'], 4), 'delivered': v['delivered'], 'via': v['via']}
            for k, v in inbox.items()}


def effective(packet):
    from . import packets as P
    return P.effective_salience(packet)


def router_project(net, pending):
    """The ROUTER's prism: pending extractions (from input or spawn triggers)
    project along the IN-EDGES of layer cells. The router has no out-edges of
    its own — it is the sum of all pathways into the network's middle layers.
    For each pending extraction e and each edge A->B (B a layer cell):
        if e.type in edge.accepts and e.type in B.accepts:
            activation(B) += e.salience_eff * edge.weight
    Returns the same inbox shape as project()."""
    inbox = {}
    for e in pending:
        for edge in net['edges']:
            target = net['cells'].get(edge['to'])
            if not target or target['role'] not in ('layer',):
                continue
            if e['type'] not in edge['accepts'] or e['type'] not in target['accepts']:
                continue  # the prism: this wavelength does not pass
            a = e.get('salience_eff', e['salience']) * edge['weight']
            slot = inbox.setdefault(edge['to'], {'activation': 0.0, 'delivered': [], 'via': []})
            slot['activation'] += a
            slot['delivered'].append({**e, 'via_edge': f"{edge['from']}->{edge['to']}", 'w': edge['weight']})
            slot['via'].append(f"{edge['from']}->{edge['to']}")
    return {k: {'activation': round(v['activation'], 4), 'delivered': v['delivered'], 'via': v['via']}
            for k, v in inbox.items()}


def who_fires(net, inbox):
    """Threshold gate. Returns [(cell_name, slot)] sorted by activation desc."""
    fires = []
    for name, slot in inbox.items():
        c = net['cells'][name]
        if c['role'] in ('judge',):  # judges never fire in forward pass
            continue
        if slot['activation'] >= c['threshold'] and slot['delivered']:
            fires.append((name, slot))
    return sorted(fires, key=lambda kv: -kv[1]['activation'])


def fallback_inject(net, inbox, packet):
    """Zero-firing hop: re-inject the strongest undelivered extraction into the
    best-weighted eligible cell (guaranteed progress). Returns (cell, slot) or None."""
    best, best_key = None, None
    for name, slot in inbox.items():
        c = net['cells'][name]
        if c['role'] == 'judge': continue
        key = slot['activation']
        if best is None or key > best_key:
            best, best_key = (name, slot), key
    return best


def snapshot(net):
    """Serializable state: weights + thresholds + update counts. This is the
    bootable artifact — a net grown here can be booted elsewhere from this JSON."""
    return {
        'format': 'jev-net-state@1',
        'cells': {n: {'threshold': c['threshold']} for n, c in net['cells'].items()},
        'edges': [{'from': e['from'], 'to': e['to'], 'accepts': e['accepts'],
                   'weight': e['weight'], 'n_updates': e.get('n_updates', 0)} for e in net['edges']],
    }


def save_state(net, path):
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(snapshot(net), f, indent=1)
        f.flush(); os.fsync(f.fileno())
    os.rename(tmp, path)
