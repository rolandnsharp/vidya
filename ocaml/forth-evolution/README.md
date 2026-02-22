# Forth-Evolution: Neural Program Synthesis with Concatenative Validation

Can a pre-trained LLM synthesize valid programs in a concatenative language,
with O(n) validation that always terminates, and evolve a growing vocabulary
through use?

## The Idea

A pre-trained 3B model proposes Forth word definitions. A Forth validator
checks them instantly (stack effect analysis — linear time, always terminates).
Valid definitions enter the dictionary. The model sees the updated dictionary
in subsequent prompts. The vocabulary grows through interaction.

This is neurosymbolic program synthesis where:
- The neural side proposes (creative, flexible, learned from data)
- The symbolic side validates (rigorous, instant, guaranteed to terminate)
- The dictionary accumulates (persistent, inspectable, composable knowledge)

## Why Forth

No other language offers this combination:

| Language | Verification          | Cost    | Can it hang? |
|----------|-----------------------|---------|--------------|
| Python   | Execute in sandbox    | Arbitrary | Yes        |
| Prolog   | Backtracking search   | Exponential | Yes      |
| Haskell  | Type inference        | Polynomial | No, but slow |
| **Forth** | **Stack depth count** | **Linear** | **Never** |

Forth validation is a single left-to-right scan: is each word in the
dictionary? Does the stack depth stay non-negative? That's it. Microseconds.
The cost of a bad proposal is effectively zero, so the model can propose
aggressively and the validator filters cheaply.

## What We Measure

- **Acceptance rate** — fraction of proposed definitions that pass validation
- **Composition rate** — do new definitions reuse previously defined words?
- **Vocabulary growth** — does the dictionary evolve or stagnate?
- **Definition quality** — do the words do what was asked?
- **Validation cost** — time per proposal vs Python/JS execution-based validation
- **Comparison baseline** — same model generating Python, where validation
  requires execution and can hang

## Architecture

```
llama.cpp (3B model)          Forth validator          Dictionary
    proposes            ────►   validates        ────►  accumulates
    definitions                 (O(n), instant)         (grows)
                                                            │
         ◄──────────────────────────────────────────────────┘
         next prompt includes the updated dictionary
```

## Setup

```bash
# Build llama.cpp
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp && cmake -B build && cmake --build build

# Download a model (Phi-3 mini Q4 or Llama 3.2 3B)

# Build this project
cd ocaml/forth-evolution
dune build

# Run
dune exec bin/main.exe -- --model path/to/model.gguf
```

## Structure

```
forth-evolution/
  lib/
    forth.ml        Forth interpreter + stack effect validator
    llama.ml        OCaml interface to llama.cpp
    evolve.ml       Evolution loop: propose → validate → accumulate
  bin/
    main.ml         CLI entry point
  stubs/
    llama_stubs.c   C FFI bindings for llama.cpp
  PLAN.md           Implementation plan
  README.md         This file
```
