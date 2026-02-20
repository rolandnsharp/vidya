# Elite Forth Repo: Optimization Guide for Claude Code

If you want the **most optimized, fastest, cleanest Forth repo possible**, and you're using **Claude Code as a development assistant**, the key isn't just code — it's **repo structure + constraints + feedback loops**.

Forth is brutally minimal. Optimization comes from **clarity, factoring, stack discipline, and measurement**, not cleverness.

Below is a battle-tested setup.

---

## 1. Pick the Right Forth First

Your repo architecture depends heavily on which Forth you target:

- **Gforth** — Best for portability, tooling, testing
- **SwiftForth** — Professional, Windows-focused
- **VFX Forth** — Extremely fast native code
- **Mecrisp-Stellaris** — Embedded Cortex-M
- **FlashForth** — Small microcontrollers

If you're optimizing for:

- **Raw speed** — VFX Forth
- **Portable clean code** — Gforth
- **Embedded minimal latency** — Mecrisp

Decide this first. It changes everything.

---

## 2. Repo Structure for Maximum Optimization

Elite layout:

```
forth-project/
│
├── src/
│   ├── core.fs
│   ├── math.fs
│   ├── memory.fs
│   ├── io.fs
│   └── platform/
│       ├── gforth.fs
│       └── vfx.fs
│
├── bench/
│   ├── microbench.fs
│   └── profiler.fs
│
├── test/
│   └── test-core.fs
│
├── build/
│   └── build.fs
│
├── docs/
│   └── design.md
│
└── README.md
```

Why this works:

- `core.fs` — pure ANS Forth only
- `platform/` — system-specific optimizations
- `bench/` — required for performance culture
- `test/` — ensures refactors don't break stack contracts

Optimization requires isolation.

---

## 3. How to Instruct Claude Code Properly

If you want Claude to produce optimized Forth, you must constrain it HARD.

Give it rules like this:

### Forth Optimization Rules

```
You are writing high-performance ANS Forth.

Constraints:
- No unnecessary stack shuffling.
- Avoid DUP SWAP ROT unless justified.
- Prefer factoring into small words (3-7 tokens).
- Avoid dynamic memory unless required.
- Inline hot paths.
- Avoid unnecessary locals.
- Comment stack effects on every word.
- Optimize for branch prediction and tail calls.
- Provide benchmark harness.
- Assume indirect threaded model unless specified.
```

That dramatically improves output quality.

---

## 4. Performance Rules That Actually Matter in Forth

### Stack Discipline = Speed

Bad:

```forth
: foo dup >r swap r> + ;
```

Better:

```forth
: foo ( a b -- a+b a )
  over + swap ;
```

Even better:

```forth
: foo ( a b -- sum a )
  tuck + ;
```

Every stack shuffle is cost.

### Avoid Deep Stack

Fast Forth words rarely exceed stack depth 3-4.

If you're hitting 6+, refactor.

### Prefer Tail-Call Style

Instead of:

```forth
: loop-test
  condition if
    recurse
  then ;
```

Prefer iterative constructs when possible.

### Avoid DOES> Unless Needed

`CREATE DOES>` is powerful but slower than plain colon definitions in many systems.

### Inline Hot Words

In Gforth:

```forth
: fast-word ( ... ) ... ; inline
```

But only for hot paths.

---

## 5. Benchmark Culture (Non-Optional)

Create a benchmark harness:

```forth
: bench ( xt -- )
  1000000 0 do
    dup execute
  loop
  drop ;
```

Then:

```forth
' my-word bench
```

Optimization without measurement is superstition.

---

## 6. Advanced Optimization Strategies

### Direct Threaded vs Subroutine Threaded

- Subroutine-threaded Forth (like VFX) benefits from:
  - Larger words
  - Reduced call overhead
- Indirect-threaded (Gforth default):
  - More factoring
  - Smaller words

Claude must know which model you're targeting.

### Memory Layout Awareness

- Align data
- Avoid misaligned fetches
- Use `CELL+` not literal numbers

### Avoid

- Unstructured control flow
- Deep return stack tricks
- Hidden state

These kill maintainability and prevent Claude from helping effectively later.

---

## 7. Give Claude This Development Workflow

When working with Claude Code:

1. Ask for pure ANS Forth first.
2. Then ask: *"Now optimize this for Gforth indirect threading."*
3. Then: *"Now reduce stack depth."*
4. Then: *"Now micro-optimize hot path only."*
5. Then: *"Add benchmark harness."*

Stepwise refinement works MUCH better than "make it fast".

---

## 8. Git + CI Setup

Use:

- Stack effect comments enforced
- Run tests in CI via Gforth
- Run benchmarks in CI (compare timings)
- Lint for stack mismatches

Even a simple GitHub Action calling Gforth is enough.

---

## 9. Philosophy of Elite Forth Repos

The fastest Forth repos share traits:

- Small words
- Flat abstractions
- Zero magic
- Clear stack effects
- Benchmarks included
- Platform separation
- Measured improvements only

Optimization is architecture, not trickery.

---

## 10. If You Want the Absolute Fastest Possible

Specify:

- Which Forth?
- Target CPU?
- OS?
- Embedded or desktop?
- What workload? (DSP? parsing? VM? numeric heavy?)

Then a **system-specific high-performance architecture** can be designed instead of generic advice.
