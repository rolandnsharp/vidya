# Principles of a Good Forth LLM Algorithm

What makes a Forth implementation of an LLM not just possible, but *right* — designed with the grain of the language rather than against it.

---

## 1. Inference Only, No Autograd

The single most important design decision. Training requires a dynamic computation graph, heap-allocated nodes, and reverse-mode differentiation — everything Forth resists. Inference requires none of it.

An inference-only LLM is:

- **Fixed topology.** The computation graph is known at compile time. Every matrix multiply, every layer norm, every attention head is a static sequence of operations. No DAG construction. No topological sort. No backward pass.
- **Predictable memory.** The KV cache is the only structure that grows, and its maximum size is bounded by the context length — known in advance.
- **Pure arithmetic.** Forward pass is just: load weights, multiply, add, apply nonlinearity. This is what Forth was born to do.

Strip away training and the problem transforms from "build a runtime system" to "orchestrate arithmetic." That is a Forth problem.

## 2. Flat, Pre-Allocated Memory Regions

No `malloc`. No arena allocators. No pointer chasing.

A good Forth LLM pre-allocates every buffer at dictionary compile time:

```forth
80 16 * CELLS ALLOT CONSTANT WTE          \ token embeddings: vocab_size x n_embd
16 16 * CELLS ALLOT CONSTANT WPE          \ position embeddings: block_size x n_embd
16 CELLS ALLOT CONSTANT X-BUF             \ current hidden state
16 CELLS ALLOT CONSTANT Q-BUF             \ query buffer
1 16 * 16 * CELLS ALLOT CONSTANT KV-CACHE \ keys+values: n_layer x block_size x n_embd
```

Every tensor is a known region at a known offset. Access is base-plus-offset arithmetic — one `+` and one `@` or `F@`. No indirection. No fragmentation. No surprises.

The principle: **if you can compute the address with arithmetic, do not store a pointer.**

## 3. Integer Quantization Over Floating Point

Forth's integer stack is its native habitat. The float stack is a guest. A good Forth LLM respects this.

Quantized inference (INT8, INT4) means:

- **Weights are small integers.** They live naturally on the data stack and in memory as single bytes or nibbles.
- **Accumulation in 32-bit integers.** Forth's `*` and `+` are native, fast, single-instruction words. No `F*`, no `F+`, no float stack depth anxiety.
- **Dequantization only at boundaries.** Convert to float only when applying softmax or the final logit scaling — not in the inner loops.
- **Bit manipulation is idiomatic.** Forth has `AND`, `OR`, `XOR`, `LSHIFT`, `RSHIFT` as first-class citizens. Packing two INT4 weights into a byte is natural, not a hack.

A quantized matmul in Forth:

```forth
: DOT-Q8 ( src1 src2 n -- acc )
  0 >R
  0 DO
    OVER I + C@S          \ signed byte from weights
    OVER I + C@S          \ signed byte from activations
    * R> + >R
  LOOP
  2DROP R>
;
```

Clean. No abstractions. The loop body is 4 words. This is the kind of code Forth rewards — tight, transparent, and close to the metal.

## 4. Words That Mirror the Architecture

Forth's power is factoring: decomposing a problem into small, named, composable words until each word is trivially correct. A good Forth LLM reads like a description of the transformer:

```forth
: EMBED       ( token pos -- )  POS-EMBED  TOK-EMBED  V+ ;
: ATTEND      ( layer -- )      DUP QKV-PROJECT  CACHE-KV  SCORE-HEADS  CONCAT-PROJECT ;
: FFN         ( layer -- )      DUP FC1  RELU  FC2 ;
: BLOCK       ( layer -- )      DUP RMSNORM  ATTEND  RESIDUAL+  DUP RMSNORM  FFN  RESIDUAL+ ;
: FORWARD     ( token pos -- )  EMBED  N_LAYERS 0 DO I BLOCK LOOP  LM-HEAD ;
: SAMPLE      ( -- token )      LOGITS>PROBS  TEMPERATURE-SCALE  WEIGHTED-PICK ;
: GENERATE    ( prompt -- )     PREFILL  BEGIN FORWARD SAMPLE DUP EMIT EOS = UNTIL ;
```

Each word is one concept. The top-level `GENERATE` reads as English. The implementation details live in the leaves. This is not just aesthetic — it is a debugging strategy. You can test `RMSNORM` in isolation at the Forth REPL, print intermediate buffers, and verify each piece before composing.

The principle: **the word hierarchy should mirror the architectural hierarchy of the transformer.** If you cannot draw a clean correspondence between your Forth words and the blocks in the model diagram, the factoring is wrong.

## 5. Buffers on the Stack, Data in Memory

A common mistake porting to Forth is trying to put tensor data on the stacks. The stacks are for *control* — addresses, indices, loop bounds, buffer pointers. The actual numbers (weights, activations) live in memory.

The stack convention for a good Forth LLM:

```
( src-addr dst-addr count -- )    \ typical word signature
```

Words take **pointers and dimensions** on the stack. They read from memory, compute, and write to memory. The stacks stay shallow (3-5 items). The data stays in flat, pre-allocated buffers.

This also means: **no allocating temporary arrays during inference.** Every intermediate result has a pre-assigned buffer. The `linear` function does not return a new array — it writes into a destination buffer whose address was passed on the stack.

```forth
: LINEAR ( src dst weight rows cols -- )
  \ For each output row, dot-product src with weight row, store in dst
  2>R
  OVER + SWAP DO          \ loop over output rows
    DUP I 2R@ DROP DOT    \ dot product: src . weight[row]
    J F!                   \ store result in dst
    2R@ DROP CELLS +       \ advance weight pointer by cols
  CELL +LOOP
  2R> 2DROP
;
```

No allocation. No return value. Just pointers in, computation, pointers out.

## 6. The KV Cache as a Circular Buffer

The KV cache is the one structure that grows during generation. A naive implementation appends to a list. A good Forth implementation uses a circular buffer with a write pointer:

```forth
VARIABLE KV-POS                          \ current write position (wraps at block_size)
: CACHE-K ( layer src -- )
  SWAP KV-K-BASE                         \ base address for this layer's key cache
  KV-POS @ BLOCK-SIZE MOD N_EMBD *       \ offset for current position
  CELLS + N_EMBD CMOVE                   \ copy n_embd cells
;
```

This gives you:
- **O(1) insertion** — write at the current position, increment, wrap.
- **No reallocation** — the buffer is fixed at `n_layer x block_size x n_embd`.
- **Natural context window** — when the position wraps, old entries are overwritten. Sliding window attention comes free.

The circular buffer is a deeply Forth-native data structure — simple, stateful, and allocation-free.

## 7. Exploit Immediacy for Debugging

Forth is an interactive language. The REPL is not a convenience — it is a design tool. A good Forth LLM leans into this:

```forth
\ Load weights, then interactively probe the model:
0 0 EMBED X-BUF 16 F.VEC              \ "What does token 0 at position 0 look like?"
X-BUF RMSNORM X-BUF 16 F.VEC          \ "What does it look like after rmsnorm?"
37 0 FORWARD LOGITS 80 F.VEC           \ "What are the logits for token 37?"
```

Every word is testable in isolation, immediately, with real weights loaded. You do not need a test harness. You do not need print statements. You type the word, the result appears. This is faster than debugging in any other language.

The principle: **design every word so it can be called at the REPL with sensible arguments.** If a word requires 12 items on the stack to test, it is doing too much. Factor it.

## 8. Memory-Mapped Weights

A good Forth LLM does not parse a model file. It memory-maps it.

Pre-process the weights once (in Python, in C, doesn't matter) into a flat binary format that matches the exact memory layout your Forth code expects — including quantization, transposition, and byte order. Then:

```forth
S" model.bin" R/O OPEN-FILE THROW
DUP FILE-SIZE THROW
SWAP MMAP                  \ or: ALLOCATE THROW DUP ROT READ-FILE THROW DROP
CONSTANT WEIGHTS-BASE
```

Now `WEIGHTS-BASE` is the base pointer. Every weight matrix is at a known offset from this base. Loading the model is one operation. No parsing. No conversion. No temporary buffers.

The principle: **move complexity out of Forth and into the build step.** The weight format should be designed so that Forth can use it with zero transformation — what's on disk is what's in memory.

## 9. Softmax as the Only Float Bottleneck

In a quantized pipeline, almost everything is integer arithmetic. Softmax is the exception — it requires `EXP`, subtraction of a maximum, and division by a sum. This is inherently floating-point.

A good Forth LLM isolates this:

```forth
: SOFTMAX ( logits-addr count -- probs-addr )
  \ 1. Find max (integer scan, then convert)
  \ 2. Subtract max and exponentiate (float)
  \ 3. Sum and divide (float)
  \ 4. Result stays in float buffer for sampling
;
```

The float stack is used in exactly one word. Everything upstream (embedding, attention scores, matmuls) is integer. Everything downstream (sampling) reads the float probabilities and converts back. The float stack never goes deeper than 3-4 items.

The principle: **quarantine floating-point operations.** Make them explicit, localized, and shallow.

## 10. Streaming Output Is Natural

Forth is a REPL language. It emits characters with `EMIT`. An LLM generates one token at a time. These are the same loop:

```forth
: CHAT ( -- )
  READ-PROMPT TOKENIZE
  PREFILL
  BEGIN
    FORWARD SAMPLE
    DUP BOS <> WHILE
    DUP TOKEN>CHAR EMIT
  REPEAT
  DROP CR
;
```

There is no buffering, no async callback, no stream wrapper. `EMIT` sends a character to the terminal the instant it is generated. The user sees tokens appear one by one. Streaming generation is not a feature — it is the default behavior.

This extends to UART, SPI, or any output device. A Forth LLM on an embedded system can stream tokens directly to a serial port with the same word. The I/O model is unified from microcontroller to terminal.

## 11. Size Discipline

Forth culture values small programs. A Forth LLM should uphold this. The inference engine — excluding weights — should be **under 500 lines**. If it is longer, the factoring is wrong or unnecessary abstraction has crept in.

This is achievable because:

- No autograd (eliminates ~40% of microgpt).
- No training loop, no optimizer (another ~30%).
- No dynamic allocation library (pre-allocated buffers).
- No data structure library (flat memory + offsets).
- Forth words are dense (one word per operation, not one line per operation).

The target:

| Component | Lines |
|-----------|-------|
| Memory layout + constants | 30-40 |
| Quantized matmul + vector ops | 40-60 |
| Attention (with KV cache) | 60-80 |
| FFN + norms | 20-30 |
| Softmax + sampling | 20-30 |
| Top-level forward + generate | 20-30 |
| I/O + tokenizer | 30-40 |
| **Total** | **~220-310** |

Small enough to read in one sitting. Small enough to fit in L1 cache on most systems. Small enough to audit by hand.

## 12. No Abstraction for Abstraction's Sake

Forth does not reward premature generalization. A `TENSOR` library with arbitrary dimensions, broadcasting, and operator overloading is a C++/Python instinct. In Forth, it is dead weight.

You do not need a general-purpose `MATMUL`. You need `ATTN-QK-MUL` that multiplies a query vector by the cached keys for one head with the exact dimensions you have. You do not need a generic `NORM`. You need `RMSNORM-16` that normalizes a 16-element vector in place.

Specialize relentlessly. Hardcode dimensions where they are known at compile time (and they are almost always known). Unroll inner loops when the trip count is 4 or 16. Let the Forth compiler (or assembler, or `CODE` words) turn these into straight-line machine instructions.

The principle: **generality is a cost, not a virtue.** Pay it only when the model architecture actually varies. For a single fixed model, specialize everything.

---

## Summary

A good Forth LLM algorithm is defined by what it *refuses* to do:

- It refuses to train. Inference only.
- It refuses to allocate dynamically. Every buffer is pre-allocated.
- It refuses to use floating-point where integers suffice. Quantize.
- It refuses to abstract beyond what the architecture demands. Specialize.
- It refuses to hide memory layout. Every byte has an address you can name.
- It refuses to grow beyond what fits in your head. Under 500 lines.

What remains is an algorithm that is transparent, fast, tiny, and — in the Forth tradition — honest about what the machine is actually doing. No runtime. No garbage collector. No framework. Just words, memory, and arithmetic.

That is the Forth way.
