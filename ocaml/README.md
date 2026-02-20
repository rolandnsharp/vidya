# MicroGPT in OCaml — Stage 1

A complete GPT implementation in pure OCaml with zero dependencies. Two versions of the same algorithm demonstrating different OCaml styles.

This is **Stage 1** of the [Vidya](../PLAN.md) project — a faithful translation of [Karpathy's microgpt.py](../microgpt.py) proving that the algorithm works in OCaml before we start optimizing.

## Quick Start

```bash
cd ocaml
ocamlopt -o microgpt_mutable 1_microgpt_mutable.ml
./microgpt_mutable

ocamlopt -o microgpt_ref 1_microgpt_ref.ml
./microgpt_ref
```

Requires OCaml (`opam install ocaml` if you don't have it). No other dependencies — pure stdlib.

Both versions train on `input.txt` (included) and generate text samples after 1000 steps. Takes a few minutes.

## The Two Files

We implemented the same algorithm twice to compare two OCaml approaches to mutable state.

### `1_microgpt_mutable.ml` — Mutable Record Fields

The autograd `value` type uses OCaml's `mutable` keyword:

```ocaml
type value = {
  id : int;
  mutable data : float;     (* read: v.data        write: v.data <- 1.0 *)
  mutable grad : float;     (* direct field mutation, like TypeScript *)
  children : value array;
  local_grads : float array;
}
```

**Style:** Imperative. Explicit `for` loops. Sequential mutation. Closest to the original Python/JavaScript versions. If you're coming from TypeScript, start here.

**Best for:** Reading and understanding the algorithm. Direct mapping to the original code. Familiar to anyone from an imperative background.

### `1_microgpt_ref.ml` — Ref Cells + Functional Style

The autograd `value` type uses `float ref` (mutable containers):

```ocaml
type value = {
  id : int;
  data : float ref;          (* read: !(v.data)    write: v.data := 1.0 *)
  grad : float ref;          (* ref cell: the field is immutable but    *)
  children : value array;    (* holds a mutable container               *)
  local_grads : float array;
}
```

**Style:** Functional. Pipe operators (`|>`), `Array.map`, `Array.fold_left`, declarative construction with `Array.init`. More idiomatic OCaml.

**Best for:** Learning OCaml patterns. Seeing how the same algorithm can be expressed functionally. Preferred style for future stages.

### Side-by-Side Comparison

Reading a value:
```ocaml
(* mutable *)  v.data
(* ref *)      !(v.data)
```

Writing a value:
```ocaml
(* mutable *)  v.data <- 1.0
(* ref *)      v.data := 1.0
```

Softmax (mutable — imperative):
```ocaml
let max_val = Array.fold_left
  (fun acc v -> Float.max acc v.data) neg_infinity logits
in
```

Softmax (ref — pipe style):
```ocaml
let max_val =
  logits
  |> Array.map (fun v -> !(v.data))
  |> Array.fold_left Float.max neg_infinity
in
```

Forward pass (mutable — for-loop):
```ocaml
let x = ref (embed_token model token_id pos_id) in
for li = 0 to n_layer - 1 do
  x := transformer_block !x model.layers.(li) ...
done;
linear !x model.lm_head
```

Forward pass (ref — fold):
```ocaml
let x = embed_token model token_id pos_id in
let x =
  Array.to_list model.layers
  |> List.mapi (fun li layer -> (li, layer))
  |> List.fold_left (fun x (li, layer) ->
    transformer_block x layer ...
  ) x
in
linear x model.lm_head
```

**Both produce identical output** — same loss, same generated text, same weights. The algorithm is the same; only the coding style differs.

## What The Model Learns

The model (5,888 parameters, 1 layer, 16-dim embeddings, 4 attention heads, 16-character context) trains on character-level text for 1000 steps using the Adam optimizer.

### Training Output

```
num docs: 16478
vocab size: 80
num params: 5888
step 1000 / 1000 | loss 1.6514
```

Loss drops from ~4.5 to ~1.65 over 1000 steps. The model is learning.

### Generated Samples

```
--- inference (new, hallucinated text) ---
sample  1: in there angerer
sample  2: aras and the il
sample  3: thice e the aler
sample  4: ad thelom Sothe
sample  5: the ore thenit e
sample  6: the at by bint i
sample  7: the the ise arow
sample  8: at ine thang the
sample  9: al thome cere is
sample 10: indy incole the
sample 11: besthe anthe t i
sample 12: Reong the ore th
sample 13: on the averere a
sample 14: the the ind the
sample 15: of the he the er
sample 16: ere the the ine
sample 17: his ar this t th
sample 18: the of ane overo
sample 19: ang the ase whis
sample 20: the of sand the
```

With only 5,888 parameters and 16 characters of context, the model has learned:
- Common English words: "the", "and", "of", "in", "this", "there"
- That spaces separate words
- Common letter patterns: "th", "er", "ing", "tion"
- That capital letters appear at the start of some sequences

It has NOT learned grammar, meaning, or coherent sentence structure — that requires orders of magnitude more parameters. This is expected and correct for Stage 1.

## For TypeScript Developers

These files are heavily commented with OCaml syntax explanations aimed at developers coming from TypeScript. Key things to watch for:

- **Float operators:** `+.` `-. ` `*.` `/.` for floats, `+` `-` `*` `/` for ints. No implicit coercion.
- **`let ... in`:** Like `const x = ...; return ...` in TS. Binds a value in a scope.
- **`|>` pipe:** `x |> f |> g` means `g(f(x))`. Data flows left to right.
- **`Array.map f arr`:** Like `arr.map(f)` in TS, but the function comes first.
- **`Array.fold_left f init arr`:** Like `arr.reduce(f, init)` in TS.
- **Pattern matching:** Like `switch` but exhaustive — the compiler warns if you miss a case.

See the "OCaml Survival Guide" comment block at the top of `1_microgpt_mutable.ml` for a full reference.

## Architecture

```
input.txt (text corpus, one doc per line)
    │
    ▼
Tokenizer (character-level: each unique char → integer ID)
    │
    ▼
Forward Pass (for each token in a document):
    Token Embedding ──► Position Embedding ──► RMSNorm
        │
        ▼
    Transformer Block (×1):
        ├── Multi-Head Attention (4 heads, dim 4 each)
        │     ├── Q, K, V projections (linear)
        │     ├── Scaled dot-product attention
        │     └── Output projection (linear)
        ├── Residual connection
        ├── MLP (linear → ReLU → linear)
        └── Residual connection
        │
        ▼
    Output Logits (linear → vocab_size scores)
        │
        ▼
    Softmax → Cross-Entropy Loss
    │
    ▼
Backward Pass (reverse-mode autodiff, chain rule)
    │
    ▼
Adam Optimizer (update all 5,888 parameters)
```

## Next Steps

This is Stage 1 of 15. See [PLAN.md](../PLAN.md) for the full roadmap.

**Stage 2:** Replace scalar autograd with a Tensor type (`float array` with shape metadata). Same algorithm, but operating on vectors/matrices instead of individual floats. This will reduce the computation graph from thousands of nodes to dozens and unlock significant speedups.
