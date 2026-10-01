"""
jev-net packets — the JEV packet is the activation carrier.

The insight this package formalizes (owner directive, wave-65):

    "JEV takes graph-of-thought, breaks it down, and filters / projects
     granular components to other cells like a prism."

A JEV packet is therefore ONE object with THREE faces:

  J — judgment   : the cell's activation semantics. confidence gates how far
                   downstream this packet's components may travel. A cell that
                   says confidence=0.2 is a neuron firing weakly; its
                   extractions arrive at the next layer scaled down.
  E — extractions: the granular components (the prism output). Each carries a
                   TYPE, a CONTENT string and a SALIENCE in [0,1]. Types are
                   the wavelengths the prism separates:
                     sub-question, entity, constraint, mechanism,
                     risk, claim, counter, anchor
  V — verdict    : the routing face. action in {route, spawn, answer, halt}.
                   'spawn' is the JEV->JEV trigger: it injects NEW components
                   into the router at the next hop — this is what makes the
                   net an ITERATOR over hops rather than a one-shot fan-out.

A packet is validated strictly; malformed packets are repaired or dropped
fail-closed, and every packet records its lineage: {cell, hop, parents} —
the parents list is the synaptic trace used by hebbian learning.
"""
import json, hashlib, time

TYPES = ('sub-question', 'entity', 'constraint', 'mechanism', 'risk', 'claim', 'counter', 'anchor')
ACTIONS = ('route', 'spawn', 'answer', 'halt')


def new_id(prefix='jev'):
    h = hashlib.sha256(f'{time.time_ns()}-{id(object())}'.encode()).hexdigest()[:12]
    return f'{prefix}-{h}'


def make(cell, hop, judgment, extractions, verdict, parents=(), notes=None):
    """Construct + validate a JEV packet. Fails closed: raises on malformed core
    fields so a broken cell can never silently poison the ledger."""
    p = {
        'jev_id': new_id(),
        'cell': cell,
        'hop': int(hop),
        'ts': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'judgment': {'verdict': str(judgment.get('verdict', ''))[:300],
                     'confidence': _clamp01(judgment.get('confidence', 0.5)),
                     'grounds': str(judgment.get('grounds', ''))[:2000]},
        'extractions': [],
        'verdict': {'action': verdict.get('action', 'route'),
                    'targets_hint': list(verdict.get('targets_hint', []))[:8],
                    'summary': str(verdict.get('summary', ''))[:2000]},
        'parents': list(parents)[:8],
        'notes': notes or {},
    }
    if p['verdict']['action'] not in ACTIONS:
        p['verdict']['action'] = 'route'
    for e in (extractions or [])[:12]:
        t = e.get('type') if e.get('type') in TYPES else 'claim'
        p['extractions'].append({'type': t,
                                 'content': str(e.get('content', ''))[:1500],
                                 'salience': _clamp01(e.get('salience', 0.5))})
    return p


def _clamp01(x):
    try: x = float(x)
    except Exception: return 0.5
    return max(0.0, min(1.0, x))


def packet_from_llm(raw_json, cell, hop, parents=()):
    """Build a packet from an LLM's JSON reply; repairs common malformations
    instead of failing the whole hop (fail-open for content, fail-closed for
    structure). Returns (packet, repairs:list[str])."""
    repairs = []
    if not isinstance(raw_json, dict):
        return None, ['non-dict reply dropped']
    j = raw_json.get('judgment') or {}
    v = raw_json.get('verdict') or {}
    if isinstance(j, str): j = {'verdict': j}; repairs.append('judgment str->obj')
    if isinstance(v, str): v = {'summary': v}; repairs.append('verdict str->obj')
    ex = raw_json.get('extractions') or []
    if isinstance(ex, dict): ex = [ex]; repairs.append('extractions obj->list')
    clean = []
    for e in ex:
        if isinstance(e, str): e = {'content': e}; repairs.append('extraction str->obj')
        if isinstance(e, dict) and e.get('content'):
            clean.append(e)
    if not j.get('confidence') and 'confidence' in j and j['confidence'] not in (None, ''):
        pass
    p = make(cell, hop, j, clean, v, parents=parents,
             notes={'repairs': repairs} if repairs else None)
    return p, repairs


def effective_salience(packet):
    """The J-face gate: an extraction leaves this cell scaled by the cell's
    confidence. Weak judgment -> weak projection. This single rule is what
    makes the network NEURAL rather than a message bus."""
    c = packet['judgment']['confidence']
    return [{**e, 'salience_eff': round(e['salience'] * c, 4)} for e in packet['extractions']]


def render_for_cell(packet, max_chars=3500):
    """Render incoming extractions as the context block a downstream cell sees."""
    lines = [f"[from {packet['cell']} @hop{packet['hop']} conf={packet['judgment']['confidence']:.2f}]"]
    for e in effective_salience(packet):
        lines.append(f"- ({e['type']} s={e['salience_eff']:.2f}) {e['content'][:280]}")
    return '\n'.join(lines)[:max_chars]


def bundle(packets, max_chars=9000):
    """Bundle several packets for a synthesizer cell."""
    return '\n'.join(render_for_cell(p, 2200) for p in packets)[:max_chars]
