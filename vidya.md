# Vidya: A Neurosymbolic Language Model with a Forth Soul

## What Vidya Is

Vidya is a small language model that thinks with two minds at once.

The first mind is a neural transformer -- a 6-layer, 128-dimensional GPT-2
style network with rotary position embeddings, weight-tied output, and
BLAS-accelerated matrix operations. It has about 1.25 million parameters.
It trains on a corpus (currently the Enneads of Plotinus) and learns what
token sequences are likely. This is the standard story. Every language model
does this.

The second mind is a Forth interpreter.

Not a toy, not a gimmick. A real Forth with a dictionary, a data stack,
stack-effect validation, and concept entries extracted from the corpus itself.
This second mind doesn't predict -- it *constrains*. It decides what's valid,
what's coherent, what's worth saying. When the neural model proposes the next
token, the Forth mind adjusts the probabilities before sampling happens.

The neural model ranks by likelihood. The symbolic system defines what's
allowed. They speak through logits.

## How It Works

### The Neural Side

Vidya's transformer is small but complete:

- **6 layers**, 128-dimensional embeddings, 8 attention heads
- **RoPE** (Rotary Position Embeddings) for position-aware attention
- **Weight tying**: the output projection shares the embedding matrix, so
  the model's concept of "what a word means" and "what word comes next" are
  the same matrix, viewed from different directions
- **Residual scaling**: each layer's output is initialized with reduced
  variance (std = 0.08 / sqrt(2L)) to keep activations bounded through
  the network
- **KV-cache** for efficient single-token inference at generation time
- **BPE tokenizer**: 500 merge rounds, ~580 token vocabulary, ~2.7
  characters per token on the Enneads

Training is Adam with cosine learning rate schedule, gradient clipping at
max norm 1.0, 400-step warmup, and 100K total steps across ~16K documents.
Standard supervised learning -- cross-entropy loss, predict the next token.

### The Symbolic Side

Five constraints are applied to the neural model's logits before sampling,
in sequence:

**1. Repetition penalty.** A 32-token ring buffer tracks recently generated
tokens. Each recent token's logit gets -1.5. Simple, effective, prevents
the model from getting stuck in loops.

**2. Word boundary detection.** When the partial word accumulated so far is
already a complete valid word, tokens that would extend it into a non-word
get penalized (-5.0). This prevents "is" + "in" from becoming "isin". The
penalty is soft -- if the neural model is very confident about a continuation,
it can override.

**3. Word validation.** Hard constraint. Tokens that would create invalid
words or invalid prefixes are masked to negative infinity. The model literally
cannot select them. The valid word and prefix sets are built once from the
corpus at startup.

**4. Concept coherence.** This is where Forth enters. When recent tokens
activate concepts in the Forth dictionary, related concepts get boosted.
If "soul" was recently generated, tokens associated with "body", "mind",
"nature" get +2.0 * 0.85^age. The boost decays over 16 tokens. The
associations come from co-occurrence statistics in the corpus, stored as
Forth concept entries with their top-8 neighbors.

**5. Topic depth penalty.** If the same concept has been activated more than
4 times, its associated tokens get penalized. This prevents the model from
circling the same cluster of ideas endlessly. Encourages thematic movement.

A safety valve: if all constraints together produce negative infinity for
every token (the model is stuck), it falls back to just the repetition
penalty. The neural model always has a way out.

### The Knowledge Layer

The Forth dictionary is populated from the corpus in a pipeline:

1. **Extract concepts**: words with frequency >= 20 and length >= 4.
   Filters out "the", "and", "is" -- keeps "soul", "nature", "body",
   "eternal".

2. **Build co-occurrence matrix**: which concepts appear in the same
   document? Count every pair.

3. **Build associations**: for each concept, take its top-8 co-occurring
   neighbors. "soul" gets ["body", "nature", "mind", "intellect", ...].

4. **Map BPE tokens to concepts**: which tokens activate which concepts?
   Exact match and prefix match. A concept becomes a Forth word of type
   `Concept` with associations, strength (normalized frequency), and
   token IDs.

5. **Populate the dictionary**: each concept enters the Forth dictionary
   alongside the primitive stack operations (DUP, +, SWAP, etc.) and any
   user-defined words.

The result: the Forth dictionary is both a symbolic reasoning substrate
and a structured knowledge base extracted from the same corpus the neural
model trained on. Two views of the same data -- one learned through
gradient descent, one extracted through counting and co-occurrence.

## Why Forth and Not Something Else

The neurosymbolic AI literature uses Prolog, Haskell, Python sandboxes,
formal verification systems. They are all more complex than necessary.

### Verification in microseconds

Forth word definitions are validated by a single left-to-right scan. For
each word in the body: is it in the dictionary? What's its stack effect?
Update the running depth. If any word is missing or the stack underflows:
reject. Otherwise: accept.

This runs in microseconds. Always terminates. No theorem prover, no type
inference, no sandbox. Compare:

| Substrate | Verification | Can it hang? |
|-----------|-------------|--------------|
| Prolog | Exponential search | Yes |
| Python | Execute in sandbox | Yes |
| Haskell | Type inference | No, but slow |
| SMT (Z3) | Constraint solving | Effectively yes |
| **Forth** | **Dictionary lookup + stack count** | **No. Never.** |

This matters because when a neural model proposes code, you want to check
thousands of candidates per second. Forth lets you do that.

### Composition is concatenation

In Forth, `A B C` is a valid program if A, B, and C are valid words. The
composition operator is whitespace. No parentheses to balance, no argument
order to learn, no nesting depth to track.

This matters for neural generation because the model's output space is
simpler. It generates sequences from a known vocabulary, separated by
spaces. Complexity grows linearly, not combinatorially.

### The dictionary is the knowledge base

Not a knowledge base bolted onto a language. The dictionary IS the language
AND the knowledge base. Words defined in terms of other words. Concepts
linked to their associations. New words can be defined that compose
existing ones. Knowledge accumulates through use.

### Forth matches the temporal structure of generation

Here's the deep argument. Text generation is sequential. Tokens arrive one
at a time, left to right. The model processes them incrementally. There is
no "whole expression" to evaluate -- just the next token and the context
so far.

Forth evaluates left to right. Each word executes as soon as it's
encountered. The stack accumulates context and releases it. This is how
generation works: incremental, temporal, with a finite working memory that
decays.

Lisp evaluates inside-out. It requires knowing the whole expression before
evaluating any of it. Lisp is spatially elegant but temporally unnatural
for token-by-token generation. Forth starts computing the moment the first
word arrives.

The stack is a natural attention mechanism. What's on top is what's attended
to right now. Push = focus. Pop = release. The 16-token concept decay in
Vidya's symbolic layer is a stack with exponential forgetting. The
correspondence is not metaphorical -- it's structural.

## What Vidya Generates

Vidya trains on the Enneads of Plotinus -- 6 treatises of Neoplatonic
philosophy. After 100K steps of training (~6 epochs), it generates text
like:

The model produces novel philosophical passages that maintain word
validity (no misspellings, no non-words), conceptual coherence (related
ideas cluster together), and thematic progression (topics evolve rather
than repeating). The symbolic constraints don't make the text robotic --
they make it *possible*. Without them, a model this small produces
unreadable fragments. With them, it produces something that reads like
a philosophical sketch.

Prompted generation works too: give it "The Soul is" and it continues
with 10 different completions, each valid, each coherent, each different.

## Where It Goes From Here

### Reinforcement Learning

Vidya currently has no way to evaluate its own output. The symbolic layer
enforces validity (are the words real?) but not quality (is the text
meaningful?). Reinforcement learning could provide this missing signal.

The natural formulation: text generation is a continuing task (no episode
boundaries). R-learning -- average-reward RL -- is designed for exactly
this. The TD error becomes r - rho + V(s') - V(s), where rho is the
running average of text quality. No discounting needed, no artificial
episode boundaries.

The Forth knowledge layer could serve as a world model for Dyna-style
planning: use the concept associations to simulate likely continuations,
evaluate multiple token sequences before committing. Interleave real
generation with model-based look-ahead.

The TD model of classical conditioning (Sutton & Barto, 1990) is directly
relevant: concept activation is a conditioned stimulus, text quality is the
unconditioned stimulus. TD learning would let Vidya learn which concept
activations predict good text -- replacing the hand-coded coherence boost
with a learned one.

### Growing Lisp Features Into Forth

Sutton wrote all his RL reference implementations in Common Lisp. Lisp has
genuine strengths -- homoiconicity, recursive data structures, pattern
matching. But the decision for Vidya is clear: keep Forth, grow the useful
parts of Lisp into it.

What to add:
- **Cons cells on the stack**: structured pairs, giving lists and trees
  without leaving Forth
- **Recursive definitions**: use the return stack with base-case checking
- **Pattern matching**: a MATCH word that destructures stack values
- **Quotations**: push a code block onto the stack as data, execute later
  (like Factor does)

What to deliberately leave out: full metaprogramming, unrestricted eval of
constructed code. Sutton's Verification Principle says the AI must be able
to check its own knowledge. Unrestricted self-modification undermines this.
Forth's verifiability is a feature, not a limitation.

### The Model Proposing Its Own Words

The current Forth dictionary is populated from corpus statistics. The next
step: let the model propose new word definitions during generation. The
symbolic layer validates them (stack-effect check, dictionary lookup).
Valid definitions enter the dictionary. The model can use them in future
output.

This is the full neurosymbolic loop: the neural model proposes, the
symbolic engine validates, the dictionary accumulates, and the model's
vocabulary grows through use. A self-evolving domain-specific language
for philosophical text, emerging from the interaction between learned
statistics and verified structure.

## The Architecture at a Glance

```
Corpus (Enneads)
    |
    v
BPE Training (500 merges) ──> Tokenizer (~580 tokens)
    |                              |
    |                    +---------+---------+
    |                    |                   |
    v                    v                   v
Neural Training    Knowledge Extraction   Symbolic Build
(100K steps)       (concepts, co-occur)   (valid words/prefixes)
    |                    |                   |
    v                    v                   v
Transformer         Forth Dictionary     Word Sets
(6L, 128d, RoPE)   (concepts + prims)   (hashtables)
    |                    |                   |
    +--------+-----------+-------------------+
             |
             v
       Generation Loop:
  1. Forward pass ──> raw logits
  2. Repetition penalty
  3. Word boundary bias
  4. Word validation (hard mask)
  5. Concept coherence (Forth)
  6. Topic depth penalty
  7. Softmax + sample
  8. Update ring buffer, partial word, concepts
  9. Emit token
             |
             v
       Generated Text
```

## The Name

Vidya (विद्या) is Sanskrit for "knowledge" or "right knowledge" -- the kind
that comes from seeing clearly, not from accumulation. In Indian philosophy,
vidya is the opposite of avidya (ignorance, misperception). It's not about
knowing more facts. It's about knowing correctly.

A language model that can tell for itself whether its words are valid, whether
its concepts cohere, whether its topics evolve -- that's reaching toward vidya.
Not there yet. But pointed in the right direction.

---

*Vidya is written in OCaml with C FFI for BLAS-accelerated matrix operations.
The source lives in `ocaml/vidya/` in the [flow](https://github.com/roland) repository.
It trains on a single CPU in about 30 minutes and generates text in real time.*
