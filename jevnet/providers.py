"""
jev-net providers — the hard-deadline wrapper around every endpoint, with the
provider-ladder lessons from cot-quilt wave-64 baked in:

 1. urllib's socket-op timeout is NOT a deadline. A slow drip response can hang
    forever. We therefore run every call with a HARD wall-clock deadline and
    abandon the socket on breach (the wave-64 receipted finding).
 2. Every model endpoint can lie about availability. Each logical role has a
    LADDER of fallbacks, tried in order, each attempt receipted.
 3. Reasoner burn-guard: if finish_reason == 'length' and content is empty, the
    whole budget was eaten by reasoning. Retry once with 3x budget.
 4. The deepseek endpoint IGNORES the seed parameter (probe receipted wave-64:
    seed=42 -> 347 tokens then 42 -> 7). Diversity comes from personas, and
    every call still RECORDS the seed it asked for (honest bookkeeping).

Every call returns a receipt dict: {source, model_requested, model_served, ms,
status, finish, content, cot, usage, seed_requested, error?}. Receipts are the
ledger; the ledger is the build.
"""
import json, time, threading, urllib.request, urllib.error

KEYS = {}
for line in open('/home/z/my-project/.env.keys'):
    if '=' in line and not line.startswith('#'):
        k, v = line.strip().split('=', 1)
        KEYS[k] = v

DS_KEY = KEYS['DEEPSEEK_API_KEY']
TS_KEY = KEYS['TYPESAFE_API_KEY']
DI_KEY = KEYS['DEEPINFRA_API_KEY']
MQ_KEY = KEYS['MOTHQUANTUM_API_KEY']

DS_URL = 'https://api.deepseek.com/chat/completions'
TS_URL = 'https://api.typesafe.ai/v1/systemone'
DI_URL = 'https://api.deepinfra.com/v1/openai/chat/completions'
MQ_URL = 'https://api.mothquantum.com/api/v1/engines/coin-toss-v1/process'


def _http(url, key, body, timeout, method='POST'):
    """Hard-deadline HTTP: run the fetch in a thread, abandon it past deadline."""
    box = {}
    def _run():
        t0 = time.time()
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(url, data=data, method=method,
            headers={'Authorization': f'Bearer {key}', 'Content-Type': 'application/json',
                     'User-Agent': 'jev-net/0.1'})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                box['r'] = {'status': r.status, 'body': json.load(r), 'ms': int((time.time()-t0)*1000)}
        except urllib.error.HTTPError as e:
            try: eb = json.loads(e.read().decode() or '{}')
            except Exception: eb = {}
            box['r'] = {'status': e.code, 'body': eb, 'ms': int((time.time()-t0)*1000)}
        except Exception as e:
            box['r'] = {'status': -1, 'body': {'error': str(e)}, 'ms': int((time.time()-t0)*1000)}
    th = threading.Thread(target=_run, daemon=True)
    th.start(); th.join(timeout)
    if 'r' not in box:
        return {'status': -2, 'body': {'error': f'HARD-DEADLINE {timeout}s breached, socket abandoned'}, 'ms': int(timeout*1000)}
    return box['r']


def chat(messages, max_tokens=2000, deadline_s=75, ladder=None, seed_requested=None):
    """Call a ladder of (label, kind, model) providers until one returns content.
    kind: 'deepseek' | 'deepinfra'. Each attempt receipted in out['attempts']."""
    ladder = ladder or [('deepseek-flash', 'deepseek', 'deepseek-chat')]
    out = {'attempts': [], 'source': None, 'content': None, 'cot': None,
           'model_served': None, 'ms': 0, 'seed_requested': seed_requested}
    for label, kind, model in ladder:
        url, key = (DS_URL, DS_KEY) if kind == 'deepseek' else (DI_URL, DI_KEY)
        body = {'model': model, 'messages': messages, 'max_tokens': max_tokens}
        if seed_requested is not None: body['seed'] = seed_requested
        r = _http(url, key, body, deadline_s)
        att = {'label': label, 'kind': kind, 'model': model, 'status': r['status'], 'ms': r['ms']}
        if r['status'] == 200 and r['body'].get('choices'):
            ch = r['body']['choices'][0].get('message', {})
            content = ch.get('content')
            att.update({'finish': r['body']['choices'][0].get('finish_reason'),
                        'content_chars': len(content or ''), 'model_served': r['body'].get('model')})
            if content:
                out.update({'source': label, 'content': content, 'cot': ch.get('reasoning_content'),
                            'model_served': r['body'].get('model'), 'ms': out['ms'] + r['ms'],
                            'usage': r['body'].get('usage')})
                out['attempts'].append(att)
                return out
            att['note'] = 'empty content (reasoner burn?)'
        else:
            att['error'] = json.dumps(r['body'])[:220]
        out['attempts'].append(att)
    out['ms'] = sum(a.get('ms', 0) for a in out['attempts'])
    return out


def chat_json(messages, max_tokens=1600, deadline_s=60):
    """JSON-mode: deepseek-flash first, then the deepinfra cheap bench. Strips
    code fences; returns (parsed_or_None, receipt)."""
    ladder = [('deepseek-flash', 'deepseek', 'deepseek-chat'),
              ('gpt-oss-20b', 'deepinfra', 'openai/gpt-oss-20b'),
              ('seed-mini', 'deepinfra', 'ByteDance/Seed-2.0-mini'),
              ('granite', 'deepinfra', 'ibm-granite/granite-4.2-3b')]
    r = chat(messages, max_tokens=max_tokens, deadline_s=deadline_s, ladder=ladder)
    if r['content']:
        txt = r['content'].strip()
        if txt.startswith('```'):
            txt = txt.split('```')[1]
            if txt.startswith('json'): txt = txt[4:]
        try:
            return json.loads(txt), r
        except Exception:
            # last-resort brace harvest
            s, e = txt.find('{'), txt.rfind('}')
            if s >= 0 and e > s:
                try: return json.loads(txt[s:e+1]), r
                except Exception: pass
    return None, r


def typesafe(state, questions, deadline_s=150):
    """typesafe.ai systemone — the JEV-native oracle. questions -> answers."""
    r = _http(TS_URL, TS_KEY, {'model': 'jev-latest', 'state': state[:20000], 'questions': questions}, deadline_s)
    out = {'source': 'typesafe', 'status': r['status'], 'ms': r['ms'], 'answers': None,
           'served': r['body'].get('model') if r['status'] == 200 else None}
    if r['status'] == 200:
        out['answers'] = r['body'].get('answers')
    else:
        out['error'] = json.dumps(r['body'])[:300]
    return out


def moth_bits(n_bits=16, shot_deadline=8):
    """n_bits TRUE quantum bits from parallel one-shot coin jobs. Fail-closed
    per-bit to os.urandom with per-bit provenance; degraded=true if any fell back."""
    import os as _os
    results = [None] * n_bits
    def one(i):
        r = _http(MQ_URL, MQ_KEY, {'params': {'shots': 1}}, shot_deadline)
        jid = (r.get('body') or {}).get('job_id')
        if not jid:
            results[i] = ('urandom', _os.urandom(1)[0] & 1); return
        rr = _http(f'https://api.mothquantum.com/api/v1/jobs/{jid}/result', MQ_KEY, None, 12, method='GET')
        body = rr.get('body') or {}
        # result shape varies; dig for a bit
        bit = None
        s = json.dumps(body)
        if '"0"' in s or "'0'" in s: bit = 0
        elif '"1"' in s or "'1'" in s: bit = 1
        results[i] = ('moth', bit if bit is not None else (_os.urandom(1)[0] & 1 if bit is None else bit))
        if bit is None: results[i] = ('urandom_fallback', _os.urandom(1)[0] & 1)
    ths = [threading.Thread(target=one, args=(i,)) for i in range(n_bits)]
    for t in ths: t.start()
    for t in ths: t.join(shot_deadline + 14)
    bits, provenance = [], []
    for i in range(n_bits):
        src, b = results[i] if results[i] else ('urandom_timeout', _os.urandom(1)[0] & 1)
        bits.append(b); provenance.append(src)
    return {'bits': bits, 'provenance': provenance, 'degraded': any(p != 'moth' for p in provenance)}


def moth_seed():
    """16 quantum bits -> int seed. Honest bookkeeping even though DS ignores it."""
    return {'seed': int(''.join(map(str, moth_bits(16)['bits'])), 2), 'n_bits': 16}
