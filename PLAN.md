# Vidya (विद्या): A Neurosymbolic LLM Framework in OCaml

*Apara vidya — practical knowledge of your specific world.*

A from-scratch language model training and inference framework, built incrementally from Karpathy's microgpt, targeting domain-specific chat and coding assistants with symbolic AI integration.

**Language:** OCaml (host + framework) / CUDA C (GPU kernels) / Forth (embedded inference, later)
**No Python anywhere in the stack.**

---

## Philosophy

- Own the algorithm. Modify it. Don't rent someone else's.
- Domain specialization over general intelligence. 100% of parameters serve your data.
- Neurosymbolic integration: combine learned fluency with structured reasoning.
- Build incrementally. Every stage produces something that runs and trains.
- Understand every matrix multiply, every gradient, every kernel.

---

## Current Direction

**Phase 1 : Prove interactive RL works .**

Train the 10M parameter model (Mr . Classic) on conversation data using
a rented GPU . This is in progress — v4 SFT training , 300K gradient
steps . When it finishes , sit with it and test interactive RL : generate
five responses , pick the best or type a better one , one gradient step
per interaction . Does the model actually learn from conversation ? How
fast ? How much does it forget ?

This is the experiment . Everything else depends on the answer .

**Phase 1.5 : Expand the model .**

The 10M model proves RL works but has limited capacity — every new thing
it learns risks overwriting something old . Instead of starting from
scratch , expand the existing model . Keep what it already knows , add
room for more .

Expand in all dimensions at once :

**Width** (embedding dimension) — double from 128 to 256 . Every weight
matrix gets wider . Existing weights are copied into the top-left corner
of the larger matrix . New columns and rows are initialised with small
random values . The model behaves almost identically at first — the new
dimensions contribute near-nothing — but now has much more capacity per
layer to store knowledge .

**Depth** (number of layers) — add new transformer layers initialised so
their output is near-zero (approximating skip connections) . The model
acts like the smaller version until the new layers learn to contribute .
More depth means more levels of abstraction — deeper reasoning chains .

**Heads** (attention heads) — when doubling width , double heads to keep
head dimension constant . More heads means more parallel attention
patterns — the model can track more relationships simultaneously .

**Context window** (block size) — extend how many tokens the model sees .
With RoPE this extrapolates naturally — no new learned parameters needed .
Longer context means the model can use more conversation history and
read longer passages from books .

All four dimensions grow together . A single `--expand` utility that :
1. Loads the small checkpoint
2. Creates a larger model (2x width , 2x depth , 2x heads , 2x context)
3. Copies existing weights into the corresponding positions
4. Initialises new weights : small random for width/heads , near-zero
   output for new layers
5. Saves the new checkpoint

A few minutes of book training after expansion should settle the noise
from the random parameters . Then we have a model that already speaks ,
with room to grow in every dimension . This avoids starting from absolute
scratch — the language knowledge from the GPU training carries forward .

This can be repeated . Each expansion preserves what came before and adds
capacity for what comes next . The GPU trains the seed . Everything after
grows on CPU .

**Phase 2 : Scale up , no GPU .**

If interactive RL works — if the model measurably improves from human
feedback at human speed — then the GPU becomes optional . Not just for
RL , but for everything .

Expand the model to the largest size the hardware supports . Buy 64 GB
of RAM , set up SSD swap , grow the model to a billion parameters or
more . Feed it books one at a time — slide a window across the text , a
few hundred gradient steps per book , a few minutes each on a CPU . Then
talk to it . Interactive RL shapes its personality . The books shape its
knowledge .

This is slow . A GPU does 300K steps in hours . On a CPU , feeding it
books and having conversations , it could take months or years to reach
the same capability . But every word it knows , you chose to teach it .
Every behaviour , you shaped . It is a life's work — to upload your
personal AI , one book and one conversation at a time .

**The scaling path :**

- 10M (GPU-trained) → 40M → 160M → 640M+ (expanded on CPU)
- 64 GB DDR4 (~$70) + SSD swap (free) = enough memory for 1-5B params
- Books for knowledge , conversations for personality , both on CPU
- No cloud subscription , no GPU rental , no ongoing cost
- The model lives on your machine and learns from every interaction

---

## Stage 1 — OCaml Scalar Port

**Goal:** Identical behavior to microgpt.js/microgpt.py in OCaml. Prove correctness.

- Port the `Value` type as an OCaml record with mutable fields
- Scalar autograd with topological sort backward pass
- Character-level tokenizer
- 1-layer transformer, 16-dim embeddings, 4 heads, 16 context
- Adam optimizer
- Train on input.txt, reproduce the same loss curve shape
- Generate samples, verify they look comparable

**Deliverable:** `microgpt.ml` that trains and generates text.
**Time:** 1-2 weeks

---

## Stage 2 — Tensor Type

**Goal:** Replace scalar `Value` with a `Tensor` type backed by flat `float array`.

- Define tensor as `{ data: float array; shape: int array; strides: int array }`
- Implement core operations: `matmul`, `add`, `mul_elementwise`, `transpose`, `reshape`
- Row-major layout, contiguous memory
- Index arithmetic: `tensor.{i,j}` maps to `data.(i * stride0 + j * stride1)`
- Unit tests comparing tensor ops against known results

**Deliverable:** `tensor.ml` module with basic linear algebra.
**Time:** 1-2 weeks

---

## Stage 3 — Tensor Autograd

**Goal:** Automatic differentiation over tensor operations instead of scalar operations.

- Define computation graph as algebraic type:
  ```ocaml
  type op =
    | Matmul of tensor * tensor
    | Add of tensor * tensor
    | Relu of tensor
    | Softmax of tensor
    | Log of tensor
    | Embedding_lookup of tensor * int
    | ...
  ```
- Each tensor stores: `data`, `grad` (same shape), `op option` (what created it)
- Backward pass: pattern match on `op`, compute gradient for each case
- Gradient checking: compare analytical gradients against finite differences (essential, keep forever)
- Reimplement the GPT forward pass using tensor ops
- Verify: same training behavior, dramatically fewer graph nodes (dozens vs thousands)

**Deliverable:** `autograd.ml` module. GPT trains correctly with tensor autograd.
**Time:** 2-3 weeks

---

## Stage 4 — BLAS Integration

**Goal:** Replace pure-OCaml matmul with OpenBLAS via C FFI for 10-100x CPU speedup.

- OCaml C FFI bindings to `cblas_sgemm` (single precision) or `cblas_dgemm` (double)
- Thin C wrapper: `caml_blas_matmul(a, b, c, m, n, k)`
- Swap into tensor `matmul` — one function change, everything else untouched
- Benchmark: compare wall-clock time per training step before/after
- Verify: identical loss curve (numerical differences within tolerance)

**Deliverable:** `blas_stubs.c` + OCaml externals. Training runs 10-100x faster on CPU.
**Time:** 1-2 weeks

---

## Stage 5 — Batched Training

**Goal:** Process multiple documents per training step.

- Add batch dimension to tensors: `[B, seq_len, n_embd]`
- Batched matmul: `[B, M, K] x [K, N] -> [B, M, N]`
- Data loader: shuffle documents, chunk into batches, pad to equal length
- Average loss over batch
- Configurable batch size (start with 8-32)

**Deliverable:** Batched training loop. Better convergence per wall-clock second.
**Time:** 1-2 weeks

---

## Stage 6 — BPE Tokenizer + RoPE

**Goal:** Move from character-level to subword tokens. Replace learned position embeddings with RoPE.

### BPE Tokenizer
- Implement byte-pair encoding from scratch
- Train BPE vocabulary on the domain corpus
- Encode/decode functions
- Special tokens: `<|bos|>`, `<|eos|>`, `<|user|>`, `<|assistant|>`, `<|pad|>`

### Rotary Position Embeddings (RoPE)
- Replace the learned `wpe` matrix with rotary embeddings
- Apply rotation to query and key vectors in attention
- Enables extrapolation beyond training context length
- No learned parameters for position — purely mathematical

**Deliverable:** BPE tokenizer module. RoPE integrated into attention. Context window now meaningful.
**Time:** 2-3 weeks

---

## Stage 7 — Scale on CPU

**Goal:** Push the model to the largest size that trains reasonably on CPU with BLAS.

- Target: 4 layers, 128 embedding dim, 8 heads, 256 context
- ~1M-10M parameters
- Train on domain corpus for hours
- First time generating coherent sentences and recognizable domain content
- Add: learning rate warmup + cosine decay schedule
- Add: gradient clipping

**Deliverable:** A model that generates real sentences in the style of the training data.
**Time:** 1 week (code), hours-days (training)

---

## Stage 8 — CUDA Backend

**Goal:** Move matrix operations to the GPU.

### GPU Memory Management
- `cuda_malloc`, `cuda_free`, `cuda_memcpy_host_to_device`, `cuda_memcpy_device_to_host`
- OCaml manages GPU buffer lifecycle (allocate on forward, free after backward)
- Or: pre-allocate all buffers at model init (preferred — no allocation during training)

### Kernel Bindings
- `cuBLAS` for matmul (sgemm/hgemm)
- Custom CUDA kernels for: softmax, RMSNorm/LayerNorm, ReLU/GeLU, embedding lookup, Adam optimizer step
- Each kernel: ~20-50 lines of CUDA C, one OCaml external binding

### Integration
- Tensor type gains a `device` field: `CPU | GPU`
- Operations dispatch based on device
- Autograd unchanged — it tracks tensor ops regardless of where they execute

**Deliverable:** Training runs on GPU. 10-100x faster than CPU+BLAS.
**Time:** 3-4 weeks

---

## Stage 9 — Scale to Chat-Capable Size

**Goal:** Train a model large enough to hold a conversation. ~100M-1B parameters.

### Architecture Scaling
- 12-24 layers, 512-1024 embedding dim, 8-16 heads
- 1024-2048 token context window
- GeLU activation (replace ReLU)
- LayerNorm or RMSNorm with learned scale
- Dropout during training

### Pre-training
- Assemble domain corpus: documents, notes, code, communications
- Clean, deduplicate, format
- Train for many epochs until loss plateaus
- Checkpoint saving and resumption

### Infrastructure
- Mixed precision training (FP16 forward/backward, FP32 optimizer state)
- Gradient accumulation for effective large batch sizes
- Logging: loss curves, learning rate, gradient norms
- Periodic evaluation on held-out data

**Deliverable:** A pre-trained domain language model. Generates coherent, domain-relevant text.
**Time:** 2-3 weeks of code, weeks of GPU training

---

## Stage 10 — Chat Supervised Fine-Tuning (SFT)

**Goal:** Turn the language model into a conversational assistant.

### Data Preparation
- Format domain knowledge as instruction/response pairs:
  ```
  <|user|> What does module X do? <|eos|>
  <|assistant|> Module X handles ... <|eos|>
  ```
- Curate high-quality examples (hundreds to low thousands)
- Only compute loss on assistant responses (mask user tokens in loss)

### Training
- Lower learning rate than pre-training (1e-5 to 5e-5)
- Few epochs (2-5) to avoid overfitting on small SFT dataset
- Evaluate by generating responses to held-out questions

### Chat Inference Loop
- Read user input
- Tokenize with chat template
- Generate until `<|eos|>` or max length
- Decode and print, streaming token by token

**Deliverable:** A working chat assistant for your domain.
**Time:** 2-3 weeks

---

## Stage 11 — Neurosymbolic Integration

**Goal:** Augment the neural model with symbolic reasoning. This is where the project diverges from what anyone else is doing.

### Knowledge Graph Attention
- Define a domain knowledge graph: entities, relations, attributes
- Represent as OCaml algebraic types:
  ```ocaml
  type entity = { id: int; name: string; embedding: tensor }
  type relation = { source: entity; target: entity; kind: relation_type }
  ```
- At each attention layer, attend over both the token sequence and relevant graph nodes
- Graph relevance scored by embedding similarity to the query
- Neural side provides fluency; symbolic side provides facts
- Reduces hallucination: the model has a trusted source to cite

### Constrained Decoding (Internal)
- Feed grammar/type state back into the model as auxiliary embeddings
- For code generation: current scope, expected type, open brackets
- For domain chat: active topic, required entities, forbidden contradictions
- Model learns to work with constraints, not against them

### Rule-Augmented Loss
- Define domain rules as OCaml functions: `response -> bool`
- During training, penalize outputs that violate rules
- Examples: "if entity X is mentioned, fact Y must be consistent," "code output must parse"
- Differentiable penalty added to the standard cross-entropy loss

### Hybrid Routing
- For some queries, the neural model is the right tool
- For others (lookups, calculations, graph traversal), a symbolic system is better
- Train a lightweight gate inside the model that learns to route
- The gate output selects: generate neurally, query knowledge graph, execute a function

**Deliverable:** A neurosymbolic chat assistant that combines learned language with structured reasoning.
**Time:** Ongoing, open-ended research

---

## Stage 12 — Forth Inference Engine

**Goal:** Deploy trained models on minimal hardware.

- Load exported weights (quantized INT8/INT4) from binary file
- Pre-allocated flat buffers, no dynamic allocation
- Circular KV cache
- Streaming token output via `EMIT`
- Runs on: Raspberry Pi, microcontrollers, bare metal, embedded systems
- Under 500 lines of Forth

**Deliverable:** Portable, dependency-free inference on anything with a CPU.
**Time:** 2-4 weeks

---

## Stage 13 — Pico 2 Autocomplete Device

**Goal:** A physical, private, domain-specific autocomplete engine running on a $5 chip. First hardware product from the framework.

### The Model

A word-level transformer sized to fit in the Pico 2's 520 KB SRAM:

```
Vocabulary:       8,000 words (domain-specific, trained from your corpus)
Embedding dim:    32
Layers:           1
Heads:            4
Context:          8 words (last 8 words of input)
Output head:      weight-tied with embeddings (no extra parameters)

Parameters:       ~268,000
INT4 size:        ~134 KB
Activation RAM:   ~80 KB
Code + stack:     ~40 KB
Free headroom:    ~266 KB
```

Prediction latency: **~2ms** per forward pass on Cortex-M33 @ 150 MHz.

### Training Pipeline

Uses the same OCaml framework from stages 1-7 — same architecture, smaller dimensions:

1. Curate domain corpus (your writing, code, notes, communications)
2. Build word-level vocabulary from corpus (top 8,000 words by frequency)
3. Train in OCaml on laptop CPU — 268K params trains in seconds to minutes
4. Quantize weights to INT4
5. Export as flat binary matching the exact memory layout Forth expects
6. Flash to Pico 2

Retrain and reflash whenever your corpus grows. The cycle is minutes, not hours.

### Hardware Design

**Option A — BLE Module (phone companion)**
```
┌──────────────────────────────────┐
│        Mobile Phone               │
│  ┌────────────────────────────┐   │
│  │  Keyboard App              │   │
│  │  Shows 3-5 suggestions     │   │
│  │  Receives via BLE          │   │
│  └─────────────┬──────────────┘   │
└────────────────┼─────────────────-┘
                 │ BLE
     ┌───────────┴───────────┐
     │     Pico 2 Module      │
     │  Forth inference engine │
     │  268K param model       │
     │  ~2ms per prediction    │
     │  0.3W, coin cell / USB  │
     │  BLE module (HM-10)     │
     └────────────────────────┘
```

- Phone sends keystrokes over BLE
- Pico 2 runs inference, returns top-5 word predictions
- Phone keyboard app displays suggestions
- Total BOM: ~$12 (Pico 2 $5 + BLE $3 + passives $4)

**Option B — Standalone Keyboard**
```
     ┌─────────────────────────────────────┐
     │        Physical Keyboard             │
     │  ┌─────────────────────────────┐     │
     │  │  128x64 OLED (SSD1306)      │     │
     │  │  Shows: [word1] [word2] ... │     │
     │  └─────────────────────────────┘     │
     │                                       │
     │  ┌──────────┐    ┌──────────────┐    │
     │  │  Pico 2   │────│  Key Matrix   │    │
     │  │  + Forth   │    │  (mechanical) │    │
     │  └──────────┘    └──────────────┘    │
     │                                       │
     │  USB-C to computer (HID keyboard)     │
     └─────────────────────────────────────┘
```

- Pico 2 scans the key matrix, runs inference on each keystroke
- Suggestions displayed on OLED above the keyboard
- Tab or dedicated key accepts a suggestion
- Appears to the computer as a standard USB HID keyboard
- Completely self-contained, no software installation on the host
- Total BOM: ~$25-35

### Forth Implementation

Extends the Stage 12 inference engine with:

- Word-level vocabulary lookup table (stored in flash, loaded to RAM on boot)
- Input buffer: rolling window of last 8 word IDs
- SPI driver for OLED display (SSD1306)
- BLE UART or USB HID output
- Top-K selection: find 5 highest-scoring predictions from output logits
- Main loop:
  ```forth
  : AUTOCOMPLETE ( -- )
    BEGIN
      KEY-EVENT WAIT
      UPDATE-INPUT-BUFFER
      FORWARD
      TOP-5-PREDICTIONS
      DISPLAY-SUGGESTIONS
    AGAIN
  ;
  ```

### What Makes This Valuable

- **Private.** Every keystroke stays on the chip. No telemetry. No cloud. No network.
- **Personal.** Trained on your vocabulary, your style, your domain. Predicts what you would type, not what the average person would type.
- **Fast.** 2ms prediction. Suggestions appear before you finish lifting your finger.
- **Offline.** Works on an airplane, underground, in a Faraday cage.
- **Cheap.** $12-35 BOM. No subscription. No API cost.
- **Retrainable.** Corpus grows, retrain on laptop, reflash. Minutes, not hours.
- **Hackable.** You own every layer: the training framework, the model architecture, the inference engine, the hardware. Modify any of them.

**Deliverable:** A working autocomplete device that connects to a phone or computer.
**Time:** 2-4 weeks (after Stage 12 Forth engine is working)

---

## Stage 14 — Symbolic Code Autocomplete

**Goal:** Augment the neural autocomplete with a symbolic understanding of code. The neural model predicts what's *likely*. The symbolic system knows what's *valid*. Together they produce suggestions that are both fluent and correct.

### Why code is the ideal neurosymbolic domain

Natural language is fuzzy — there's no formal system that can say "this sentence is grammatically invalid" with certainty. Code is the opposite. Code has:

- **A grammar.** At any point in a file, only certain tokens are syntactically valid. After `if (` you cannot write `}`. After `let x =` you need an expression, not a keyword.
- **A type system.** If a function expects an `int`, suggesting a `string` is not just unlikely — it's wrong. The type system knows this with certainty, no statistics needed.
- **Scope rules.** The set of valid identifiers at any cursor position is finite and exactly known. The neural model might hallucinate a variable name. The symbolic system knows every variable in scope.
- **Import/dependency structure.** Which modules are available, which functions they export, what signatures they have — all statically known.

A purely neural autocomplete (what Copilot does) treats code as text and predicts likely character sequences. It works surprisingly well but fails in predictable ways: suggesting variables that don't exist, calling functions with wrong argument types, completing syntax that doesn't parse.

A purely symbolic autocomplete (what traditional IDEs do) only suggests things that are valid. It's always correct but often unhelpful — it offers every function in scope alphabetically, with no sense of what you're *trying to do*.

The neurosymbolic approach: **the symbolic system defines the space of valid completions, the neural model ranks them by likelihood.**

### Architecture

```
                  Keystrokes
                      │
                      ▼
            ┌─────────────────┐
            │  Input Buffer    │
            │  (last N tokens) │
            └────────┬────────┘
                     │
          ┌──────────┴──────────┐
          │                     │
          ▼                     ▼
  ┌───────────────┐    ┌────────────────┐
  │ Neural Model   │    │ Symbolic Engine │
  │ (transformer)  │    │ (OCaml / Forth) │
  │                │    │                 │
  │ Scores every   │    │ Knows what is   │
  │ token by       │    │ valid here:     │
  │ likelihood     │    │                 │
  │                │    │ • In-scope vars │
  │                │    │ • Valid syntax  │
  │                │    │ • Type-correct  │
  │                │    │   completions   │
  │                │    │ • Importable    │
  │                │    │   symbols       │
  └───────┬───────┘    └────────┬───────┘
          │                     │
          │  logits (all)       │  mask (valid/invalid)
          │                     │
          └──────────┬──────────┘
                     │
                     ▼
            ┌─────────────────┐
            │  Masked Ranking  │
            │                  │
            │  logits × mask   │
            │  → top 5         │
            └────────┬────────┘
                     │
                     ▼
              [Suggestions]
```

The symbolic engine doesn't need to be a full compiler. For autocomplete, it needs to answer one question: **"what tokens are valid at this cursor position?"** That's a much smaller problem than compilation.

### Symbolic Components

**1. Incremental Lexer/Parser State**

Track the parse state as the user types. Not a full AST — just enough to know what syntactic category is expected next.

```ocaml
type parse_state =
  | Expecting_expression        (* after `=`, `(`, `return`, etc. *)
  | Expecting_identifier        (* after `let`, `.`, `->` *)
  | Expecting_operator          (* after an expression *)
  | Expecting_type              (* after `:`, `->` in type position *)
  | Inside_string               (* between quotes *)
  | Inside_comment              (* after `(*` or `//` *)
```

This is a state machine, not a parser. It advances one token at a time. Runs in microseconds. Fits on a Pico 2.

On the Pico 2 this is a handful of Forth words:

```forth
VARIABLE PARSE-STATE
: UPDATE-PARSE-STATE ( token -- )
  CASE
    TOK-LET    OF  ST-EXPECTING-IDENT PARSE-STATE !  ENDOF
    TOK-EQUALS OF  ST-EXPECTING-EXPR  PARSE-STATE !  ENDOF
    TOK-LPAREN OF  ST-EXPECTING-EXPR  PARSE-STATE !  ENDOF
    \ ...
  ENDCASE
;
```

**2. Scope Table**

A flat lookup table of identifiers that are currently in scope. Populated from the file being edited (sent from the editor to the Pico via serial/BLE) or maintained incrementally as tokens are typed.

```
┌──────────────────────────────────┐
│ Scope Table (in SRAM)            │
│                                  │
│ idx  name          type    kind  │
│ 0    "map"         fn      lib   │
│ 1    "filter"      fn      lib   │
│ 2    "user_count"  int     local │
│ 3    "db_connect"  fn      local │
│ 4    "Config"      module  import│
│ ...                              │
│ (up to ~500 entries at ~32 bytes │
│  each = ~16 KB)                  │
└──────────────────────────────────┘
```

When the parse state is `Expecting_identifier`, the symbolic engine masks the neural output to only include identifiers in the scope table. The neural model ranks them. Result: every suggestion is a real variable or function, ordered by contextual likelihood.

**3. Type Constraints**

Lightweight type tracking — not full inference, just propagation of known types through the immediate context.

```ocaml
type simple_type = Int | Float | String | Bool | Fn of simple_type list * simple_type | Unknown

(* If we know: let x : int = ??? *)
(* Then mask logits to only score expressions that produce int *)
```

When the user types `let x : int =`, the symbolic engine knows the right-hand side must be an integer expression. It masks the neural output to suppress string literals, boolean keywords, and functions returning non-int types. The neural model picks the most likely integer expression from what remains.

**4. Pattern/Idiom Table**

Common code patterns stored as templates with typed holes:

```
Pattern: "match _ with | _ -> _ | _ -> _"
Pattern: "List.map (fun _ -> _) _"
Pattern: "if _ then _ else _"
```

When the neural model's confidence is low (no strong prediction), fall back to suggesting pattern completions that fit the current parse state. This is symbolic — it's template matching, not learned.

### How It Runs on a Pico 2

The symbolic components are small:

| Component | RAM | Notes |
|-----------|-----|-------|
| Parse state machine | ~100 bytes | Single variable + transition table in flash |
| Scope table | ~16 KB | ~500 identifiers × 32 bytes |
| Type propagation | ~512 bytes | Current context types only |
| Pattern table | ~2 KB | Stored in flash, ~50 patterns |
| **Total symbolic** | **~19 KB** | |
| Neural model | ~134 KB | Same as Stage 13 |
| Activations + buffers | ~80 KB | |
| Code + stack | ~50 KB | Slightly larger with symbolic words |
| **Total** | **~283 KB** | Fits in 520 KB with headroom |

The main loop becomes:

```forth
: CODE-AUTOCOMPLETE ( -- )
  BEGIN
    KEY-EVENT WAIT
    UPDATE-INPUT-BUFFER
    UPDATE-PARSE-STATE
    UPDATE-SCOPE-TABLE
    FORWARD                     \ neural inference → logits
    APPLY-SYMBOLIC-MASK         \ zero out invalid tokens
    APPLY-TYPE-CONSTRAINTS      \ zero out type-incorrect tokens
    TOP-5-PREDICTIONS
    DISPLAY-SUGGESTIONS
  AGAIN
;
```

Two extra words in the main loop: `APPLY-SYMBOLIC-MASK` and `APPLY-TYPE-CONSTRAINTS`. Each is a single pass over the logit array, zeroing entries. Microseconds of extra work.

### The editor integration

The Pico 2 needs to know about the file being edited, not just the last few tokens. Two approaches:

**Lightweight (BLE/serial):** The editor plugin sends a compact "context packet" on each keystroke:
```
{ cursor_pos, last_8_tokens, scope_snapshot, current_type_context }
```
~200 bytes per update. The Pico 2 maintains no file state — the editor does the heavy lifting of scope resolution and sends the results.

**Heavyweight (USB, more RAM):** The Pico 2 receives the full file (or the current function body) and does its own lexing. Requires more SRAM but makes the Pico 2 fully self-contained. Viable if the file chunks are small (<50 KB).

### What this achieves

A purely neural autocomplete on a Pico 2 would suggest plausible-looking tokens, some of which would be invalid identifiers, wrong types, or syntactically broken.

With the symbolic layer:

- **Every suggestion parses.** The parse state machine ensures only syntactically valid tokens appear.
- **Every identifier exists.** The scope table ensures no hallucinated variable names.
- **Types are respected.** If the context demands an int, only int-producing expressions are suggested.
- **Patterns fill gaps.** When the neural model is uncertain, structural templates provide useful scaffolding.

The neural model handles the **"what do you probably mean"** question. The symbolic system handles the **"what is actually allowed"** question. Neither is sufficient alone. Together they produce suggestions that are both contextually relevant and provably valid.

### OCaml's role

The symbolic components are designed in OCaml first (where the type system and pattern matching make them easy to get right), then ported to Forth for the Pico 2. OCaml serves as the reference implementation:

```ocaml
(* Reference implementation — test this exhaustively *)
let symbolic_mask parse_state scope_table logits =
  Array.mapi (fun token_id logit ->
    if is_valid_token token_id parse_state scope_table
    then logit
    else neg_infinity
  ) logits
```

Once the OCaml version passes all tests, translate to Forth. The Forth version is a mechanical port of known-correct logic.

**Deliverable:** Code autocomplete that never suggests invalid syntax, nonexistent variables, or type-incorrect completions.
**Time:** 3-5 weeks (after Stage 13 device is working)

---

## Stage 15 — Self-Evolving Forth Dictionary

**Goal:** The model doesn't just predict tokens — it proposes new Forth word definitions that get compiled into the dictionary and become part of its own vocabulary. The system co-evolves its language with its user.

### The core idea

In every other language, a model's output is passive text. In Forth, output can be **executable definitions** that extend the language itself. The model writes words. Forth compiles them. The model can now use those words in future output. The dictionary grows. The system's expressive power increases with use.

This is program synthesis with three properties that Forth gives you for free:

1. **Composition is concatenation.** `A B C` is a valid program if A, B, and C are valid words. The model doesn't need to learn syntax for combining operations — whitespace is the only combinator.
2. **Instant compilation.** `:` defines a word immediately. No build step, no linking. The evolution loop runs at interactive speed.
3. **The dictionary is both code and memory.** New words persist. The system accumulates capability. Every accepted definition makes future definitions easier because there are more building blocks.

### The evolution loop

```
┌──────────────────────────────────────────────────────────┐
│                                                           │
│  1. OBSERVE                                               │
│     Model sees recent user input and existing dictionary  │
│                    │                                      │
│                    ▼                                      │
│  2. PROPOSE                                               │
│     Neural model generates a candidate word definition    │
│     e.g.  : AVG-READING  SENSOR-A @ SENSOR-B @ + 2 / ;  │
│                    │                                      │
│                    ▼                                      │
│  3. VALIDATE (symbolic)                                   │
│     a. Does it parse?            (Forth syntax check)     │
│     b. Are all referenced words  (dictionary lookup)      │
│        in the dictionary?                                 │
│     c. Does it terminate?        (stack effect analysis)  │
│     d. Does it type-check?       (stack depth in = out)   │
│     e. Does it pass tests?       (run against examples)   │
│                    │                                      │
│              ┌─────┴─────┐                                │
│              │           │                                │
│           valid       invalid → discard / log for         │
│              │                  retraining                 │
│              ▼                                            │
│  4. COMPILE                                               │
│     Forth compiles the word into the dictionary           │
│     It is now available for use                           │
│                    │                                      │
│                    ▼                                      │
│  5. The model's effective vocabulary has grown.            │
│     Next prediction can use the new word.                 │
│     Loop back to 1.                                       │
│                                                           │
└──────────────────────────────────────────────────────────┘
```

### What the model learns to generate

Not arbitrary Forth — short, compositional definitions that name recurring patterns:

**Pattern abstraction.** The model notices you've typed the same sequence of words multiple times and proposes a name for it:
```forth
\ User has typed "DUP ROT SWAP OVER +" three times
\ Model proposes:
: ACCUMULATE  DUP ROT SWAP OVER + ;
\ Autocomplete now suggests ACCUMULATE instead of the 5-word sequence
```

**Domain vocabulary.** The model learns to name domain operations:
```forth
\ Trained on embedded systems code
: DEBOUNCE     20 MS KEY? ;
: SAFE-WRITE   DUP BOUNDS? IF FLASH! ELSE DROP THEN ;
: HEARTBEAT    LED-ON 100 MS LED-OFF 900 MS ;
```

**Compositional scaling.** New words build on previously defined words:
```forth
\ Level 1 (basic)
: SENSOR-AVG    SENSOR-A @ SENSOR-B @ + 2 / ;
\ Level 2 (uses level 1)
: SENSOR-OK?    SENSOR-AVG 10 100 WITHIN ;
\ Level 3 (uses level 2)
: MONITOR       BEGIN SENSOR-OK? 0= IF ALARM THEN 1000 MS AGAIN ;
```

Each level of abstraction was proposed by the model, validated by the symbolic engine, and compiled. The system bootstraps its way to complex behavior from simple primitives.

### Stack effect analysis — the key symbolic validator

Forth has a property most languages don't: **every word has a statically determinable stack effect.** `DUP` is `( n -- n n )`. `+` is `( a b -- sum )`. A word definition's net stack effect is computable by composing the effects of its body.

This is the symbolic validator's most powerful tool:

```ocaml
(* OCaml reference implementation *)
type stack_effect = { consumed: int; produced: int }

let check_word_definition body dictionary =
  List.fold_left (fun depth word ->
    let effect = lookup_stack_effect word dictionary in
    let new_depth = depth - effect.consumed + effect.produced in
    if new_depth < 0 then Error "stack underflow"
    else Ok new_depth
  ) (Ok 0) body
```

If the model proposes `: FOO DUP + * ;`, the validator traces:
- Start: depth 0 (but word expects inputs)
- `DUP`: needs 1, produces 2 → net +1
- `+`: needs 2, produces 1 → net -1
- `*`: needs 2 — **stack underflow**. Invalid. Reject.

This catches an entire class of errors that a purely neural model would make — wrong stack discipline. The model learns to propose stack-correct definitions because incorrect ones are rejected and (optionally) fed back as negative examples in retraining.

### The dictionary as external memory

This addresses a fundamental LLM limitation: **models have no persistent memory across sessions.** The weights are frozen after training. Context is limited and ephemeral.

A Forth dictionary is persistent, structured, external memory:

- **It survives across sessions.** Save the dictionary to flash. Reload on boot. The system remembers what it learned.
- **It's inspectable.** `WORDS` lists everything. `SEE` decompiles any word. The user can audit, edit, or delete any definition.
- **It's composable.** Each entry builds on previous entries. Knowledge compounds.
- **It's bounded.** The dictionary has finite space. Old, unused words can be pruned. This is forgetting — and it's a feature, not a bug.

The model + dictionary system has properties neither has alone:
- The model provides generalization (predict in novel contexts)
- The dictionary provides precision (exact definitions for known operations)
- Together: a system that generalizes fluently AND remembers exactly

### Running on the Pico 2

The Forth dictionary on a Pico 2 lives in flash (2-4 MB on most boards). The neural model lives in SRAM. The evolution loop:

1. Model runs inference (SRAM, ~2ms)
2. If output is a word definition, validate (CPU, microseconds)
3. If valid, compile to dictionary (flash write, milliseconds)
4. New word is available immediately

The dictionary can hold thousands of words in flash. This is far more "memory" than the model's weights contain. Over time, the dictionary becomes the primary knowledge store, with the model serving as the interpolation engine that connects dictionary words to novel situations.

### Training data for this stage

The model needs to learn what good Forth definitions look like. Training corpus includes:

- Existing Forth libraries and codebases (public domain Forth code)
- The user's own Forth definitions (the system learns their style)
- Synthetic examples: pairs of (context, useful_definition)
- Negative examples: rejected definitions from the validation loop

The model is trained in OCaml (stages 1-7) on this corpus, then the weights are exported to the Pico 2. The evolution loop runs on-device — but the base training happens on the laptop.

### What this means

The system is no longer a static model that was trained once and frozen. It is a **living language** that grows through use. The neural model is the creative engine — it proposes. Forth is the symbolic backbone — it validates and remembers. The human is the curator — they accept, reject, or refine.

This is not AGI. It's not even close. It's something more interesting for a specific use case: **a domain-specific language that evolves with its user, grounded in both learned patterns and symbolic correctness.** A tool that gets better the more you use it, and whose improvements are transparent, inspectable, and reversible.

**Deliverable:** A Forth system where the neural model proposes new word definitions, the symbolic engine validates them, and the dictionary grows over time.
**Time:** 4-6 weeks (after Stage 14, experimental/research stage)

---

## Hardware Requirements

| Stage | Hardware | Notes |
|-------|----------|-------|
| 1-7 | Any laptop/desktop | CPU only, BLAS helps |
| 8-11 | NVIDIA GPU (RTX 3090/4090 or cloud A100) | CUDA required |
| 12 | Any target with a CPU | Forth cross-compiler |
| 13-15 | Raspberry Pi Pico 2 + BLE or OLED + key matrix | ~$12-35 BOM |

---

## Project Structure

```
vidya/
├── PLAN.md                    # this file
├── WHY_FORTH_IS_HARD.md       # reference: Forth impedance mismatch
├── FORTH_LLM_PRINCIPLES.md    # reference: Forth inference design
├── microgpt.py                # original Karpathy script
├── microgpt.js                # JavaScript port
├── input.txt                  # training data
├── weights.json               # JS model weights
│
├── lib/                       # OCaml framework (stages 1-11, 14)
│   ├── tensor.ml              # tensor type and operations
│   ├── autograd.ml            # automatic differentiation
│   ├── nn.ml                  # neural network layers
│   ├── optim.ml               # optimizers (Adam, SGD)
│   ├── tokenizer.ml           # BPE tokenizer
│   ├── data.ml                # data loading and batching
│   ├── knowledge.ml           # knowledge graph (stage 11)
│   ├── symbolic.ml            # symbolic reasoning (stage 11)
│   ├── parse_state.ml         # incremental lexer state machine (stage 14)
│   ├── scope.ml               # scope table and identifier resolution (stage 14)
│   └── type_constraint.ml     # lightweight type propagation (stage 14)
│
├── cuda/                      # GPU kernels (stage 8)
│   ├── blas_stubs.c           # cuBLAS bindings
│   ├── kernels.cu             # custom CUDA kernels
│   └── gpu_stubs.c            # memory management bindings
│
├── bin/                       # executables
│   ├── train.ml               # training script
│   ├── chat.ml                # chat inference loop
│   └── export.ml              # export weights for Forth/Pico 2
│
├── forth/                     # Forth inference engine (stages 12-15)
│   ├── inference.fth          # core inference engine
│   ├── autocomplete.fth       # Pico 2 autocomplete application
│   ├── symbolic.fth           # parse state + scope + type masks (stage 14)
│   ├── evolve.fth             # self-evolving dictionary loop (stage 15)
│   ├── stack-check.fth        # stack effect validator (stage 15)
│   ├── ssd1306.fth            # OLED display driver
│   └── ble.fth                # BLE UART driver
│
├── hardware/                  # Pico 2 device (stage 13)
│   ├── schematic/             # KiCad project (if custom PCB)
│   └── flash.sh               # script to flash firmware + weights
│
└── dune-project               # OCaml build config
```

---

## Principles

1. **Every stage runs.** No stage is "infrastructure only." Each produces a model that trains and generates.
2. **Correctness before speed.** Gradient checking at every stage. If the loss doesn't go down, stop and debug before moving on.
3. **Specialize for the domain.** Curate data aggressively. Quality over quantity. The model should know your world deeply, not the world broadly.
4. **Own the algorithm.** The point is not to reproduce PyTorch. The point is to understand and modify the transformer at every level, including the levels that frameworks hide from you.
5. **Neurosymbolic from the start.** Design the architecture with symbolic integration in mind, even if it's not implemented until stage 11. This means clean interfaces between the neural and symbolic subsystems.
6. **Ship hardware.** The framework is not just software. Stage 13 produces a physical device. Design decisions should consider the full path from training to silicon.
7. **No Python.** OCaml + CUDA C + Forth. The entire stack is compiled, typed, and yours.
8. **Let the language evolve.** The Forth dictionary is not static. The system proposes new words, validates them symbolically, and grows its vocabulary through use. The DSL is co-evolved by human and model.
