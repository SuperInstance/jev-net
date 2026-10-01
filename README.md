# jev-net

**A neural network whose neurons are JEV calls.**

Not a network that *uses* JEV somewhere in a pipeline — a network where the
JEV packet IS the activation carrier, the prism projection IS the synaptic
transfer, and the JEV→JEV spawn trigger IS the recurrent iterator.

## The three faces of a JEV packet

Every fired cell emits one object with three faces:

| Face | Field | Neural analogue |
|------|-------|-----------------|
| **J** — judgment | `confidence` in [0,1] | presynaptic firing strength: everything this cell projects downstream is scaled by it |
| **E** — extractions | typed granular components (`sub-question`, `constraint`, `mechanism`, `risk`, `claim`, `counter`, `entity`, `anchor`), each with `salience` | the prism output: one judgment separated into wavelengths |
| **V** — verdict | `action ∈ {route, spawn, answer, halt}` | the trigger: `spawn` injects NEW components into the router at the next hop (JEV→JEV), making the net an iterator, not a one-shot fan-out |

## The transfer function

```
activation(B) += extraction.salience × A.confidence × edge.weight
     per edge A→B, per extraction whose type passes both filters
```

- **salience** — how load-bearing the component is (prism separates wavelengths)
- **confidence** — the presynaptic gate (weak judgment → weak projection)
- **weight** — the synaptic strength (the learned parameter)

Cell B **fires** iff `activation(B) ≥ threshold(B)` and it received ≥1
delivered extraction. If a hop would fire nobody, the router re-injects the
strongest pending component into the best-weighted eligible cell — guaranteed
progress, receipted as `fallback_inject`.

The router is not a neuron; it is **the sum of all pathways into the middle
layers**: pending extractions project along every in-edge of every layer cell
whose `accepts` filter matches. This one decision (v0.1 bug #2, receipted) is
what makes spawn-triggered components routable anywhere in the net.

## Learning (hebbian credit assignment)

After each forward pass, a judge model (deepseek-v4-pro, Qwen3.5-397B fallback)
labels each cell's contribution: `pivotal +2 / useful +1 / neutral 0 / noise −1`
plus an overall score and named gaps.

Per edge that actually delivered into a credited cell (share ∝ weight):

```
w += η × share × delta/2        η=0.12, clamped to [0.05, 1.0]
```

Thresholds adapt too: a cell that earned +2 fires easier next time
(×0.97), a noise cell fires harder (×1.03), clamped [0.05, 0.6].

Only pathways that carried real signal get credit — `skeptic→builder` stayed
at 0.7 in the first learn because the prism delivered no skeptic-wavelength
components into builder that pass. No phantom credit.

Judge output is fuzzy-matched back to canonical cell names (the judge echoes
`[builder @hop1]` / truncates `mechanism`→`echanism`; subsequence matching
recovers it). Judging models are unreliable narrators; the ledger is not.

## Learned state (first pass, run fwd-213130-16c5, judge score 9/10)

```
input→mechanism  0.80 → 1.00   (pivotal ×2)
input→skeptic    0.70 → 0.77   (useful ×2)
mechanism→builder 0.75 → 0.79  (pivotal)
builder→skeptic  0.50 → 0.52
mechanism threshold 0.12 → 0.113 (earned excitability)
```

State snapshots live in `nets/state.json` (`jev-net-state@1` format) — a net
grown here boots elsewhere from that JSON alone. This is the saved-state,
bootable-by-others deliverable shape.

## Recurrence observed

skeptic and mechanism both re-fired at hop 2 with accumulated activations
(3.87, 3.70 vs 1.17, 1.23 at hop 1) via the `mechanism→builder→skeptic` and
`builder→skeptic` loops — the net behaves like a recurrent network, not a
DAG. Hop cap (`max_hops`) + spawn budget bound the iteration.

## Provider ladders (wave-64 lessons, receipted)

- Hard wall-clock deadlines; urllib socket-op timeouts are not deadlines.
- Every cell has a ladder `[(label, kind, model)...]`; attempts are receipted.
- Truncated JSON (`finish=length`) is recovered by brace-balance repair, then
  one repair-retry at 2× token budget. Two truncations recovered in run 2.
- deepseek ignores `seed` (wave-64 probe); mothquantum coin-toss supplies 16
  TRUE quantum bits per run for honest seed bookkeeping.
- Judge output is normalized; the ledger is the build.

## Layout

```
jevnet/providers.py   hard-deadline HTTP, ladders, typesafe, moth bits
jevnet/packets.py     JEV packet schema, validation, repair, salience math
jevnet/graph.py       cells/edges, router prism, threshold gate, state snapshots
jevnet/forward.py     the forward pass (hops, spawns, synth, route check)
jevnet/learn.py       judge + hebbian update + fuzzy cell matching
run.py                CLI: forward / learn / show / probe
nets/default.json     starter topology + weights
nets/state.json       learned state (bootable artifact)
runs/                 per-run receipts (atomic writes)
```

## Usage

```bash
python run.py forward "How should a small project structure its plugin API?" --no-typesafe
python run.py learn runs/fwd-XXXXXX-XXXX          # judge + hebbian update
python run.py show nets/state.json                # inspect learned weights
```
