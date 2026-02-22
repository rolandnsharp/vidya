# Forth-Evolution Implementation Plan

## Goal

Demonstrate that a pre-trained LLM can synthesize valid Forth programs,
validated in O(n) time (always terminates), and that the accumulated
definitions form an evolving vocabulary the model reuses.

## Phase 1: llama.cpp FFI bindings

Same approach as symbolic-eval. Minimal OCaml bindings to llama.cpp:
- Load a GGUF model
- Tokenize text, feed tokens, get logits
- Sample tokens (temperature sampling)
- Detokenize back to text

Can share stubs with symbolic-eval or keep independent. Independent is
simpler for now — deduplicate later if both experiments mature.

### OCaml interface

```ocaml
module Llama : sig
  type model
  type context

  val load_model : string -> model
  val create_context : model -> context
  val free : context -> unit

  val generate : context -> string -> float -> string
  (* generate ctx prompt temperature → response text *)
end
```

We might not need raw logit access for this experiment. The model generates
text (Forth definitions), and we validate the text. If we just need string
in / string out, we could even shell out to ollama's CLI as a first pass
and replace with FFI later.

## Phase 2: Forth interpreter

Bring back forth.ml from vidya (git history) or rewrite. We need:

- Dictionary (hashtable of named entries with stack effects)
- Primitives (DUP, DROP, SWAP, +, -, *, /, etc.)
- Stack effect validation (the key feature)
- Definition compilation (: NAME body ;)
- Execution (for testing definitions)
- Dictionary introspection (WORDS, SEE — for including in prompts)

The validator is the critical piece:
```ocaml
val validate : dictionary -> string list -> (stack_effect, string) result
(* Always terminates. O(n) in body length. *)
```

## Phase 3: Evolution loop

The core experiment. Each iteration:

1. **Build prompt** — include current dictionary state (word names + stack
   effects), the task description, and a few examples of valid definitions

2. **Generate** — ask the model to propose a Forth word definition

3. **Parse** — extract `: NAME body ;` from the model's response

4. **Validate** — run stack effect analysis on the body
   - All words in dictionary? If no → reject, log which word was unknown
   - Stack underflow? If yes → reject, log where
   - Accept → compile into dictionary

5. **Log** — record the proposal, validation result, timing, and dictionary
   state after each iteration

6. **Repeat** — next iteration sees the updated dictionary

### Prompt design

The prompt is critical. The model needs to:
- Know what words are available (dictionary listing)
- Know the stack effect notation (examples)
- Know the task (what to define)
- See examples of valid definitions

```
You are a Forth programmer. Here are the words currently in the dictionary:

  DUP ( n -- n n )    duplicate top of stack
  DROP ( n -- )        discard top of stack
  SWAP ( a b -- b a )  swap top two
  + ( a b -- sum )     add
  - ( a b -- diff )    subtract
  * ( a b -- prod )    multiply
  SQUARE ( n -- n² )   previously defined

Define a Forth word that computes the cube of a number.
Respond with only the definition in the format: : NAME body ;
```

### Task progression

Start simple, get harder:
1. SQUARE (DUP *)
2. CUBE (DUP DUP * *)
3. ABS (DUP 0 < IF NEGATE THEN) — requires control flow
4. MAX (OVER OVER < IF SWAP THEN DROP) — two inputs
5. Compositions using previously defined words
6. Open-ended: "define a useful word" — does the model compose creatively?

## Phase 4: Comparison baseline

Same tasks, but ask the model to generate Python functions instead.

For each Python proposal:
- Execute in a subprocess with timeout
- Check if it runs without error
- Check if it produces correct output on test cases

Measure:
- Validation time (Forth: microseconds. Python: milliseconds + subprocess)
- Hang rate (Forth: 0%. Python: non-zero — infinite loops possible)
- Acceptance rate (fraction of valid proposals)
- Lines of code per definition (Forth should be shorter)

## Phase 5: Analysis

### Core metrics

1. **Acceptance rate over time** — does it improve as the dictionary grows?
   The model sees more examples and more available words.

2. **Composition depth** — do later definitions use earlier ones?
   Plot: for each accepted definition, how many of its body words are
   user-defined (not primitives)?

3. **Vocabulary growth curve** — cumulative accepted definitions over
   iterations. Linear = steady learning. Plateau = model hit its limit.
   Accelerating = model benefits from accumulated vocabulary.

4. **Validation cost comparison** — wall-clock time per validation,
   Forth vs Python. Should be orders of magnitude.

5. **Failure analysis** — why do proposals fail? Unknown words? Stack
   underflow? Syntax errors? This reveals what the model struggles with.

### Stretch metrics

6. **Cross-session persistence** — save dictionary, reload, continue.
   Does the model benefit from a pre-built dictionary?

7. **Dictionary quality** — are the definitions actually correct?
   Run each on test inputs and check outputs.

## Phase 6: Write-up

Paper structure:
1. Intro: neurosymbolic program synthesis, verification gap problem
2. Key insight: concatenative languages have O(n) validation
3. System: llama.cpp + Forth validator + evolution loop
4. Experiments: acceptance rate, composition, growth, cost comparison
5. Results
6. Discussion: implications for neural code generation

## Dependencies

- llama.cpp (shared library build)
- A GGUF model (Phi-3 mini Q4 or Llama 3.2 3B)
- OCaml 5.x, dune

## Open Questions

- How much Forth does Phi-3 / Llama 3.2 actually know? Might need few-shot
  examples in every prompt.
- Should we fine-tune? If the base model's Forth knowledge is too weak,
  a small LoRA fine-tune on Forth code could help. But that changes the
  claim from "pre-trained model" to "fine-tuned model."
- Control flow: the initial Forth has no IF/THEN/ELSE. How far can the
  model get with just stack manipulation and arithmetic?
- How to handle the model generating invalid syntax (missing colon,
  missing semicolon, extra text)? Regex extraction should be robust.
