# RepoLens

RepoLens indexes a git repository (Python-only for the MVP) into Postgres and
exposes it for navigation / symbol search / an agent orchestrator. Spec lives in
[`docs/RepoLens_SDD (1).md`](docs/RepoLens_SDD%20%281%29.md) (mind the space + parens in the
path). Backend is FastAPI + SQLAlchemy; UI is React; Postgres + Ollama via
`docker-compose.yml`.

## Running backend tests / scripts

From `backend/` use the project venv: `.venv/Scripts/python`. Scripts and tests
bootstrap `sys.path` themselves (see `tests/conftest.py`), so either set
`PYTHONPATH=.` or just run from `backend/`. DB default is
`postgresql+psycopg2://repolens:repolens@localhost:5432/repolens` (override via
`DATABASE_URL` / `.env`).

## Pipeline status (SDD §7)

- **Environment + schema (§7 steps 1–2, §11): done, committed.** Alembic migration
  `backend/alembic/versions/0001_initial_schema.py` creates all 10 §11 tables;
  verified against live Postgres.
- **File walker — `app/indexing/walker.py`: done, verified.** Walks a repo root,
  prunes VCS/venv/build dirs, writes one `files` row per in-scope `.py` file
  (path POSIX-relative to root, `language`, `loc`, `last_modified`). **Does not
  commit** — caller owns the transaction. Verified: 83 files for flask.
- **Parser — `app/indexing/parser.py`: done, verified.** `parse_file` (pure,
  tree-sitter-python) → `index_file_symbols` (persists, flushes for method→class
  FK, **does not commit**) → `parse_and_index_file` (glue). Verified: 918 symbols
  for flask. **Scope is deliberate:** top-level classes + functions, plus
  methods one level into a class body. **Nested functions are excluded by design**
  (implementation detail, not addressable). Confirmed correct against multiple
  real flask files, including `@overload`-decorated duplicates. Docstrings = first
  statement of a block iff it's an `expression_statement` of only `string` literals.
- **Import graph — `app/indexing/import_graph.py`: done, verified.** `parse_imports`
  (pure, tree-sitter; whole-tree walk so `if TYPE_CHECKING:` / `try/except
  ImportError` / function-local imports surface — *opposite* scope posture from the
  symbol parser) → `ImportResolver` (built once from the repo's file rows; maps
  absolute dotted names via an importable-name map + src-layout handling, and
  relative imports via path-space resolution) → `index_file_imports` (persists
  **one `imports` edge per file→module dependency**, does not commit).
  Verified: 467 edges for flask (187 internal, 280 external), manually
  cross-checked against `src/flask/app.py`, confirmed correct. **Known limitation
  (§18):** PEP 420 namespace packages (dirs with no `__init__.py`) are not honored
  by the absolute resolver; flagging rather than guessing is the mandated posture.
- **Reference index — `app/indexing/reference_index.py`: done; flask batch run
  pending verification (blocked by the Bash classifier outage → user runs).**
  `find_references` (pure wrt DB — shells out to `ripgrep` for a symbol's
  whole-word name across the repo's `.py` files, drops the symbol's own
  `def`/`class` binding line, caps at `DEFAULT_CAP=50` via `--max-count`) →
  `index_symbol_references` (resolves hits via a caller-built `path -> file_id`
  dict, persists **one `references` edge per distinct referencing `file`**, does
  not commit). Pure/persist split matches the other stages; `_parse_ripgrep_output`
  is the no-ripgrep/no-DB core the unit tests pin. **Scope is a name-based
  reference graph, NOT a resolved call graph (SDD §8/§13):** whole-word textual
  matches are recorded even when a *different* symbol merely shares the name;
  comments/string literals are NOT filtered (documented trade-off — reliable
  filtering needs a tokenizer, same "flag rather than guess" posture as the
  import graph's PEP 420 note). Tests: `tests/test_reference_index.py`.

## Agent layer (SDD §9)

- **Planner agent — `app/agents/planner.py`: done, verified.** Built standalone,
  pre-LangGraph (SDD §9.1): given a user question + a description of the three MVP
  tools (`search_symbols`/`search_files`/`search_text`, SDD §10), asks the local
  LLM via Ollama and returns a structured JSON plan in the §9.1 shape
  `{"steps":[{"agent":"search","tool":…,"args":{…}}]}`. stdlib `urllib` only — no
  `ollama`/`httpx`/`langchain-ollama` dep; decoupled from `app.db` (reads its own
  `OLLAMA_HOST`/`OLLAMA_MODEL`). Demo `scripts/planner_demo.py` feeds 5 diverse
  Flask questions in raw + json-forced modes: **10/10 correct structural plans**,
  qwen3:8b via Ollama, ~2.8s/call.
- **Key finding — Ollama `think:false` is required, not optional.** Without it, Qwen3
  sometimes burns its entire `num_predict` budget on chain-of-thought and returns an
  *empty* `message.content` (`done_reason="length"`) — looks like a JSON-parse
  failure but is really budget exhaustion; the built-in retry never rescues it
  (re-thinking re-empties the budget). The `/no_think` prompt directive alone made
  it *worse* (4/5 empty) — the fix must be the native Ollama `"think": false` payload
  param (top-level, **not** under `options`), not a prompt trick. `think=False` is
  also ~4x faster. Defaults to `False` in `plan()`/`_chat_payload`, exposed as a
  param so a future agent role doing harder reasoning can pass `think=True`.
  Confirmed in `scripts/planner_diag.py`.
- **Search Agent — `app/agents/search_agent.py`: done, verified.** Built standalone,
  pre-LangGraph (SDD §9.2): a pure, **no-LLM dispatcher** — unlike the Planner, it
  *executes* tools rather than choosing them. Given one plan step (the
  `{"tool":…,"args":…}` shape the Planner produces), it validates the tool name is
  one of the three MVP search tools, **injects** the orchestrator-supplied
  `repository_id` + `Session` (the Planner is never allowed to put those in
  `args`), forwards only the whitelisted optional kwargs (`kind`→`search_symbols`,
  `regex`→`search_text`, with `regex` bool-coerced to survive a weak model emitting
  `"true"`), and returns the tool's structured hits wrapped in a consistent
  `SearchStepResult{tool, repository_id, args, hits}` the Synthesizer will consume.
  Raises `SearchDispatchError` (a `ValueError` subclass) **clearly** — naming the bad
  tool and listing valid ones — on unknown/malformed tool name, non-object step/args,
  or a missing `query`; never swallowed into a silent `[]` (empty hits ARE a valid
  answer, not a dispatch failure). Reproduces the Planner's `KNOWN_TOOLS`/
  `OPTIONAL_TOOL_ARGS` as a `SEARCH_TOOLS` registry so the two modules can't drift.
  Tests: `tests/test_search_agent.py` — **13/13 pass** (10 pure
  dispatch/coercion/helper tests + 3 live dispatch-vs-flask, one per tool).
- **Planner → Search Agent chain — verified clean.** Throwaway
  `scripts/planner_search_chain_demo.py` runs the full §9.1→§9.2 hop end-to-end for
  3 flask questions spanning all three tools: calls `plan()` (`fmt="json"`,
  `think=False`, `temperature=0`), takes the **first plan step unmodified** straight
  into `build_search_step_result()`, and prints both stages plus an
  `args (planner)`-vs-`args (used)` diff that flags any DROPPED/COERCED kwarg.
  **3/3 dispatched cleanly with zero friction** — unmodified steps route to the
  correct tool, real hits returned, no arg-name mismatch, no unexpected type. (The
  SDD §9.1 *example* writes `{"name": …}` while the real tools take `query`; the
  Planner's production tool-description already uses the real arg names, so the two
  line up rather than collide — and the dispatcher rejecting a missing `query` would
  have surfaced it as `SearchDispatchError`, so any drift is loud not silent.)
- **Synthesizer agent — `app/agents/synthesizer.py`: done, verified.** Built
  standalone, pre-LangGraph (SDD §9.4), same pattern as Planner/Search: the ONLY
  user-facing NL agent. Consumes the `SearchStepResult` list the Search Agent emits
  (one per executed plan step) + the original question, makes one Ollama
  `/api/chat` call, returns `SynthesizerResult{answer, raw_response, hit_file_paths,
  cited_file_paths, wall_time_s}`. The SDD §9.4 grounding rule ("cite only facts
  present in its structured inputs") lives in the system prompt. **No
  `format="json"`** (prose — no schema to satisfy, no parse to rescue, so the SDD
  §18 JSON lever doesn't apply and retries aren't wired); `think=False` +
  `temperature=0` defaults mirror the Planner's lessons (budget-exhaustion-to-empty
  fix, ~4x faster); `num_predict=700` for a cited prose answer. `\r\n` / lone-`\r`
  in docstrings + matched source lines are normalized to `\n` at the boundary
  where text *enters* agent context (`_normalize_newlines`) — not a file sweep; the
  deferred cosmetic sweep of `app/agents/search_agent.py` (`\r\n` docstrings) +
  `app/agents/planner.py` / `app/tools/text_search.py` (pre-existing mojibake `§`
  → `Â§`) remains deferred, harmless (boundary-normalize already fixes the behavior
  the CRLF artifact could cause). Demo `scripts/synthesizer_demo.py` runs
  Planner→Search→Synthesizer for the same 3 flask questions and prints the first
  real user-facing cited answer + a citation eyeball aid (ground vs cited, invented
  vs uncited); the full verification build is deferred — the result carries BOTH
  sets so it's a later one-liner (`set(cited) - set(hit)`). Config imported from the
  Planner (one owner of defaults, can't drift); stdlib `urllib` transport; raises
  the same `OllamaError`.
- **Known limitation (Synthesizer — Should-Have polish, NOT a blocker):** at
  `temperature=0` with qwen3:8b, when several hits share the EXACT SAME symbol name
  the Synthesizer reliably cites only ONE of them, dropping the others. Confirmed
  **0/5** for "cite both" via `scripts/synth_q1_repeat.py` (Q1 = "where is the
  Blueprint class defined" — two real `Blueprint` defs in `flask/blueprints.py` and
  `flask/sansio/blueprints.py`; every run cited only the sansio one). A
  same-name-coverage instruction was added to the system prompt and **reverted** —
  it stayed 0/5, so this is a hard, deterministic capability limit at temp=0 / this
  model size, NOT a prompt-wording issue; further prompt iteration is not the lever
  (the code comment above the prompt says so and guards against re-adding it).
  **Zero hallucination is unaffected**: every citation is still always grounded in
  a real hit (no invented path), confirmed across 11 total demo + diagnostic runs.
  Potential future fixes — temperature tuning, two-pass synthesis, a larger model
  — are out of scope for now.
- **Context Agent — `app/agents/context_agent.py` + four tools
  (`app/tools/{_path_resolve,architecture,dependency_graph,file_history,github_metadata}.py`):
  verified, clean-green end-to-end — 74/74 passing across
  `tests/test_context_tools.py` + `tests/test_context_agent.py`, including live
  dispatch against the real indexed flask repo (the 32 DB-backed tests previously
  blocked by the Bash-outage handoff now run green).** Built standalone,
  pre-LangGraph (SDD §9.3), the
  same pattern as the Search Agent: a pure, **no-LLM dispatcher** that *executes*
  the four context tools rather than choosing them (the §9.3 structural/relational +
  git/issue-metadata role). Given one plan step (the
  `{"agent":"context","tool":…,"args":{…}}` shape the Planner will emit), it
  validates the tool name, **injects** `repository_id` + `Session`, extracts
  `target` positionally (the tools' shared 2nd arg), forwards only the
  whitelisted optional knobs as kwargs, and returns the tool's structured result
  wrapped in a `ContextStepResult{tool, repository_id, args, result}` the
  Synthesizer consumes. Raises `ContextDispatchError` (a `ValueError` sublist)
  **clearly** on unknown/malformed tool/step/args AND on a missing REQUIRED
  `target` (the instance-keyed tools); an unresolvable `target` (empty result)
  is a *valid* answer, NOT a dispatch error. Unlike the search tools' shared
  `query`, the context tools have **heterogeneous required args** — modeled in a
  `CONTEXT_TOOLS` registry (name → callable + required-set + optional-set), with
  `CONTEXT_REQUIRED_ARGS`/`CONTEXT_OPTIONAL_ARGS` derived from it (single source
  of truth) for the future Planner-menu convergence. The four tools (SDD §10):
  - **`_path_resolve.py`** (shared internal helper, underscore-prefixed — not
    agent-callable): resolves a plan `target` (file path OR dotted module name) to
    a `file_id` in 4 ranked stages — exact path → dotted-as-module (`<path>.py`
    tail) → dotted-as-package (`<__init__.py>` tail) → unique substring; ties →
    `None` ("don't guess"). Shared by the two graph + two metadata tools.
  - **`dependency_graph.py`** — directed **forward + backward closures** up to
    `depth` hops (NetworkX `single_source_shortest_path_length` over the internal
    `imports` edges + their reverse). `depth≤0` = node-only; `depth=1` = direct
    in/out; `depth≥2` = genuinely transitive (the in/out closures are
    *independent* — a file imported-by-many but importing-nothing reports
    empty-out / populated-in honestly; the old undirected-BFS design collapsed
    this and made `depth>1` vestigial, so it was rewritten). Default `depth=1`.
  - **`architecture.py`** — NetworkX in-degree centrality ("key files") +
    package-dir module map + (focused) a radius-bounded induced subgraph;
    `scope="whole"` when no target, `"file"` when focused. **Package boundaries
    must be resolved repo-wide (from all `files` paths), NOT from the
    radius-bounded subgraph** — otherwise a package's `__init__.py` outside the
    focus radius causes every focused file's module to silently collapse to `""`
    (bug found + fixed: `package_dirs` was derived per-subgraph inside
    `_module_map`, now built once from the full `path_by_id`). Reuses the
    resolver's `_package_dirs`/`_enclosing_module`. Defaults `top_k=10`, `radius=2`.
  - **`file_history.py`** — `commits`+`file_commits` → contributors (count desc,
    name-asc tiebreak) + recent commits (newest-first, capped). **Backing PENDING
    (§7 step 7):** the `commits`/`file_commits` tables exist (migration 0001) but
    aren't populated yet; returns `last_modified` (walker §7 step 1, done) with
    empty lists until then. Proven by synthetic-DB tests; a live-flask test pins
    the clear-empty shape (and will *fail* on purpose the moment §7 step 7 lands,
    signaling the test + docstring update — the same pinned-to-fail-when-fixed
    posture as the reference-index cap-pinner).
  - **`github_metadata.py`** — `issues` rows partitioned into issues/PRs by the
    URL heuristic (`/pull/`→PR, else issue; `None`/ambiguous→issue) — documented
    because the §11 `issues` table has no type discriminator column; optional
    `linked_files` membership filter. **Backing PENDING (§7 step 8):** the
    `issues` table is empty until the PyGithub indexer lands; clear-empty until
    then. Synthetic-DB tests prove the split + filter; a live-flask test pins it.
  - Both graph tools read the live `imports` edge slice; both metadata tools read
    live `files`/`commits`/`issues` tables. All split pure-core / injected-Session
    wrapper, owning no transaction (the project convention). **Tests:**
    `tests/test_context_tools.py` (pure cores + synthetic-DB resolver/history/
    metadata + live-flask graph + clear-empty-until-§7) and
    `tests/test_context_agent.py` (pure malformed-step/registry/`_forward_kwargs`
    + live dispatch of all 4 tools, incl. kwarg-forwarding + clear-empty). **Demo:**
    `scripts/context_agent_demo.py` hand-constructs 5 plan steps (Planner menu not
    yet wired — see below) spanning all 4 tools + both architecture scopes,
    dispatches against flask, prints per-tool `dataclasses.asdict` structured
    results (SDD §10 JSON shape).
- **Context Agent — two deferred follow-ups, neither a blocker:**
  (1) **Planner context-menu wiring.** The verified Planner is untouched; teaching
  it to EMIT `agent:"context"` steps needs its own Ollama re-verification (a new
  tool menu in `DEFAULT_TOOLS_DESCRIPTION`, new `KNOWN_TOOLS`/`OPTIONAL_TOOL_ARGS`
  entries, a re-run of the 5-question demo). Until then the Context Agent is
  exercised by hand-constructed steps; the SDD §9.1 example's `"file"` arg-name vs
  the tool's `"target"` is reconciled in that follow-up. (2) **§7 step 7 + step 8
  backing.** Git-history (`commits`/`file_commits`) and PyGithub (`issues`)
  indexing lights up the two metadata tools' lists untouched — their code + tests
  are already correct; only the empty lists populate (the pinned clear-empty tests
  will then fail-by-design, prompting their one-line update).

## Test repo

- **flask**, cloned at `C:\Users\Atharv Sharma\Desktop\Work\flask`.
- **`repositories.id = 2`** in the DB (status `indexing`).

## Throwaway driver scripts (`backend/scripts/`)

Not part of the app — one-shot harnesses to exercise a pipeline stage against the
real flask repo before the real orchestrator (`pipeline.py` + async job / API
endpoint) is wired up. Existing ones:
- `run_walker_once.py` — walker → `files` (clears repo's file rows first for idempotency).
- `parse_single_file_poc.py` — pure tree-sitter parse+print, no DB.
- `index_all_flask_symbols.py` — parse + `symbols` for all 83 flask `files` rows;
  per-file SAVEPOINT isolation, clears repo's symbols first, one commit, reports
  ok/failed-per-error/total-symbols-with-DB-ground-truth.
- `index_all_flask_imports.py` — `parse_imports` + `index_file_imports` for all 83
  flask `files` rows; builds `ImportResolver` once, per-file SAVEPOINT isolation,
  clears repo's `imports` edges first, one commit, reports ok/failed/edges-created
  (resolved-vs-external) with a DB-ground-truth count check.
- `index_all_flask_references.py` — `find_references` + `index_symbol_references`
  for all 918 flask `symbols` rows (joined to their `file` for definition path +
  line); builds `path -> file_id` once, per-symbol SAVEPOINT isolation, clears
  repo's `references` edges first, one commit, reports ok/failed/edges-created /
  hits-dropped / symbols-with-zero-refs / **symbols that hit the cap** with a
  DB-ground-truth count check.
- `planner_search_chain_demo.py` — Planner (§9.1) → Search Agent dispatcher (§9.2)
  end-to-end chain: for 3 flask questions (spanning all 3 tools), calls `plan()`
  (`fmt="json"`/`think=False`/`temperature=0`), takes the **first plan step
  unmodified** into `build_search_step_result()`, prints both stages + an
  `args (planner)`-vs-`args (used)` diff that flags DROPPED/COERCED kwargs; catches
  `SearchDispatchError`/`OllamaError`/tool errors as labeled FRICTION. **3/3 clean,
  zero friction**.
- `run_search_agent_tests.py` — throwaway pytest runner for
  `tests/test_search_agent.py` (the Bash-classifier-outage workaround the env quirk
  below describes). 13/13 pass.
- `synthesizer_demo.py` — Planner (§9.1) → Search Agent (§9.2) → Synthesizer (§9.4)
  end-to-end chain demo: for the SAME 3 flask questions as
  `planner_search_chain_demo.py`, runs `plan()` (`fmt="json"`/`think=False`/
  `temperature=0`), takes the first plan step unmodified into
  `build_search_step_result()`, passes the `SearchStepResult` (as a one-element
  list) to `synthesize()`, and prints the first real user-facing cited answer + a
  citation eyeball aid (ground vs cited, invented vs uncited; the full verification
  build is deferred). Every raise is caught as labeled FRICTION, never swallowed.
- `synth_q1_repeat.py` — diagnostic that runs Q1 ("where is the Blueprint class
  defined") through the full chain N=5 times and reports, per run, whether the
  answer cites the non-sansio `blueprints.py`, `sansio/blueprints.py`, BOTH, or
  neither (citation detected by full path OR 2-segment tail, since the two files
  share the basename and only the parent dir disambiguates them). Established the
  0/5 "cite both" hard-failure result recorded in the Synthesizer known-limitation
  note above — the evidence the same-name prompt rule was reverted.
- `context_agent_demo.py` — Context Agent dispatcher (§9.3) demo, pre-Planner-menu:
  hand-constructs 5 plan steps in the Planner's shape
  (`{"agent":"context","tool":…,"args":{…}}` — the Context Agent is a no-LLM
  dispatcher, so this skips the LLM `plan()` call) spanning all 4 tools + both
  architecture scopes, dispatches each against the flask repo via
  `dispatch_context_step(step, repo_id, session)`, and prints per-tool legible
  summaries + the full `dataclasses.asdict` structured result (SDD §10 JSON shape).
  Skips the 2 graph tools (not the whole run) if flask's import graph isn't
  indexed; reports the 2 metadata tools' clear-empty-until-§7 shape. Catches
  `ContextDispatchError`/tool errors as labeled FRICTION; closes with a summary
  table (expect `dispatched: 5/5 friction: 0` against a fully-indexed flask).

Same pattern each time: `sys.path` bootstrap, re-root each POSIX-rel `files.path`
under `REPO_PATH` to read from disk.

## Env quirk: Bash classifier outages

The Bash tool's safety classifier has intermittent outages: execution fails with
`claude-opus-4-8[1m] is temporarily unavailable, so auto mode cannot determine the
safety of Bash right now`. Read-only tools (Read/Grep/Glob) are unaffected. When
this hits, don't loop-retry — **write the script to disk and have the user run it
directly in PowerShell** (or via the `!` prompt prefix), and paste the output back.

## Next up

- **All four §9 agents are now built standalone (pre-LangGraph): Planner (§9.1),
  Search Agent (§9.2), Context Agent (§9.3), Synthesizer (§9.4).** The
  Planner→Search chain is proven (3/3 clean); the Context Agent is now verified
  clean-green end-to-end (74/74, incl. live flask dispatch) — only its two
  follow-ups remain (Planner context-menu wiring; §7 step 7+8 backing);
  Synthesizer was already verified. The next
  discrete steps, in rough order:
  - **Wire the Planner's context menu** (the Context Agent follow-up above) —
    teach the verified Planner to EMIT `agent:"context"` steps (new
    `DEFAULT_TOOLS_DESCRIPTION` entries for the 4 tools, new `KNOWN_TOOLS`/
    `OPTIONAL_TOOL_ARGS` entries, reconcile the SDD §9.1 `"file"` arg-name to
    the tools' `"target"`), re-run the 5-question demo for its own Ollama
    re-verification. Then a `planner_context_chain_demo.py` mirroring
    `planner_search_chain_demo.py` proves the §9.1→§9.3 hop on real steps.
  - **Wire the full LangGraph orchestration (SDD §15)** — the §9.1 Planner picks
    search OR context steps, the §9.2/§9.3 dispatchers execute them, the §9.4
    Synthesizer consumes both `*StepResult` shapes into one cited NL answer. The
    four agents are the nodes; this is the connective tissue + persistence
    (`chat_messages.tool_trace`, SDD §11).
- **Verify the reference index against flask.** Run
  `scripts/index_all_flask_references.py` (blocked by the Bash classifier
  outage → user runs it) and confirm its report, then lift the "pending
  verification" caveat in the reference-index pipeline-status bullet above.
- **Then git history + GitHub metadata (SDD §7 later steps) — now also the backing
  for the Context Agent's two metadata tools.** The core indexing stages — walker,
  parser, import graph, reference index — are all in place; the remaining §7 steps
  add commit/author/blame history (step 7 → `file_history`'s lists) and repo
  metadata via PyGithub (step 8 → `github_metadata`'s issues/PRs). Their code +
  tests already ship correct; landing the backings flips the pinned clear-empty
  tests to fail-by-design, prompting their one-line update.
