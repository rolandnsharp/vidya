# Symbolic-Eval Implementation Plan

## Goal

Test whether Vidya's symbolic constrained decoding improves output from a
pre-trained 3B parameter model. If yes, write it up.

## Phase 1: llama.cpp FFI bindings

Minimal OCaml bindings to llama.cpp via C stubs. We don't need the full API —
just enough to load a model, feed tokens, and get logits back.

### Functions needed

```c
// Model lifecycle
llama_model * llama_model_load(const char * path, ...);
llama_context * llama_new_context(llama_model * model, ...);
void llama_free(llama_context * ctx);
void llama_model_free(llama_model * model);

// Inference
void llama_decode(llama_context * ctx, llama_batch batch);
float * llama_get_logits(llama_context * ctx);

// Tokenizer
int llama_tokenize(llama_model * model, const char * text, int * tokens, int max);
const char * llama_token_to_str(llama_model * model, int token);
int llama_n_vocab(llama_model * model);

// Sampling (we do our own, but need this for baseline comparison)
int llama_sample_token(llama_context * ctx, ...);
```

### OCaml interface

```ocaml
module Llama : sig
  type model
  type context

  val load_model : string -> model
  val create_context : model -> context
  val free_context : context -> unit
  val free_model : model -> unit

  val tokenize : model -> string -> int array
  val token_to_string : model -> int -> string
  val n_vocab : model -> int

  val eval : context -> int array -> unit    (* feed tokens *)
  val get_logits : context -> float array    (* get logit array *)
end
```

### C stubs (stubs/llama_stubs.c)

Thin wrappers that convert between OCaml values and llama.cpp types.
Same pattern as the BLAS stubs in vidya — `CAMLparam`, `CAMLreturn`,
`caml_copy_double`, etc.

### Build

Link against `libllama.so` from the llama.cpp build. Dune `c_library_flags`
and `c_flags` point to the llama.cpp build directory.

## Phase 2: Symbolic layer integration

Copy `symbolic.ml`, `knowledge.ml`, and `utils.ml` from vidya. These are
self-contained — no dependencies on vidya's tensor/model/forward code.

Adapt the vocabulary handling:
- Vidya uses its own BPE vocab (string array, index = token ID)
- llama.cpp has its own tokenizer with `llama_token_to_str`
- Build the vocab array from llama.cpp's tokenizer at startup

The symbolic layer doesn't care where the vocab came from. It just needs
`vocab : string array` mapping token IDs to strings.

Knowledge building needs a corpus — same `chat_input.txt` or any domain text.

## Phase 3: Generation loop

```
load model
build concept knowledge from corpus
for each prompt:
  tokenize prompt
  eval prompt tokens (prefill)
  loop:
    get logits
    apply symbolic constraints (or not, for baseline)
    sample token
    eval [token]
    record token for TD update
  end
```

Two modes:
- `--baseline` — sample directly from model logits (temperature only)
- `--symbolic` — apply full symbolic layer before sampling

Same prompts, same temperature, same seed. Compare outputs.

## Phase 4: Evaluation

### Automated metrics
- **Repetition rate:** fraction of 4-grams that appear more than once
- **Vocab validity:** fraction of generated words found in corpus vocabulary
- **Perplexity difference:** does symbolic biasing hurt perplexity on held-out text?

### Manual evaluation
- Blind A/B comparison on 50-100 prompt pairs
- Rate: coherence, relevance, repetitiveness, naturalness

### TD learning curve
- Run 500+ generations with TD learning active
- Plot association weight changes over time
- Show that weights converge (learning) vs random walk (noise)

## Phase 5: Write-up

If results are positive:
- Clean, testable claim: symbolic post-processing improves pre-trained LLM output
- No retraining required — inference-time only
- Lightweight — adds milliseconds per token
- Framework-agnostic — works on any model that exposes logits

## Dependencies

- llama.cpp (build from source, need shared library)
- A GGUF model file (Phi-3 mini Q4 recommended — small, good quality)
- Training corpus for knowledge building (chat_input.txt or domain text)
- OCaml 5.x, dune, same toolchain as vidya

## Open Questions

- Does the BPE tokenizer difference matter? Vidya's vocab is built from our
  corpus; the pre-trained model has its own. The symbolic layer operates on
  token IDs via vocab string mapping, so it should work — but edge cases
  around subword boundaries might need handling.
- How much does corpus choice affect concept quality? Should we use the
  model's training data distribution, or the target domain?
- Is the TD learning signal strong enough on a model that's already good?
  The reward is raw logit confidence — a 3B model is already confident on
  most tokens, so deltas might be small.
