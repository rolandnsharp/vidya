# Why Porting microgpt to Forth Is Painfully Slow

A comprehensive analysis of the architectural mismatches between a scalar autograd GPT and the Forth programming language.

## Executive Summary

microgpt is a ~200-line script that packs an extraordinary amount of implicit complexity: a heap-allocated DAG of autograd nodes, dynamic nested data structures, higher-order array operations, and recursive graph traversal. Every one of these patterns fights against Forth's fundamental design: a stack-oriented, manual-memory, word-at-a-time language with no native support for objects, closures, or dynamic allocation. The port is not conceptually difficult — it is mechanically brutal.

---

## 1. The Computation Graph Is a Heap-Allocated DAG

This is the core problem. The `Value` class is not just a number — it is a node in a directed acyclic graph:

```javascript
class Value {
  constructor(data, children = [], localGrads = []) {
    this.data = data;
    this.grad = 0;
    this._children = children;     // references to other Value nodes
    this._localGrads = localGrads; // parallel array of floats
  }
}
```

Every arithmetic operation (`add`, `mul`, `exp`, `log`, `pow`, `relu`) allocates a **new** Value node that points back to its operands. A single forward pass through one token creates **thousands** of these nodes. The backward pass then does a topological sort over the entire graph and walks it in reverse.

In JavaScript/Python, this is trivial — the garbage collector handles allocation and cleanup, references are implicit pointers, and arrays resize automatically.

In Forth:

- There is no garbage collector. You must manually allocate and free every node.
- There is no native concept of a "struct with fields." You must define memory layouts by hand and compute byte offsets.
- Each Value node needs: 1 float (data), 1 float (grad), a variable-length list of child pointers, and a parallel variable-length list of gradient floats. Variable-length fields in a manually-managed struct are a nightmare.
- The graph is created dynamically during the forward pass and varies in size per input. You cannot statically pre-allocate it.
- You must implement your own arena/pool allocator or risk fragmenting the dictionary.

A realistic Forth implementation needs a custom memory allocator just to get started, before writing any neural network code.

## 2. Variable-Length Arrays Are Everywhere

The codebase is saturated with dynamically-sized lists:

| Usage | Example |
|-------|---------|
| Children per Value node | `[this, other]` or `[this]` — 1 or 2 elements |
| KV cache per layer | `keys[li].push(k)` — grows by one vector per token |
| Attention logits | Length equals number of tokens seen so far |
| Loss accumulation | `losses.push(...)` — one per position |
| Tokenized input | Varies per document |
| Vocabulary | Derived from dataset at runtime |

Forth has no dynamic arrays. You have:

- The **data stack** (typically 32-256 cells deep — not viable for arrays of thousands of elements).
- The **return stack** (even smaller, and misusing it corrupts control flow).
- **ALLOTted memory** in the dictionary (fixed at compile time).
- **ALLOCATEd memory** (ANS Forth optional word set — essentially raw `malloc` with no structure).

Every dynamic list must be manually implemented as either a linked list (pointer chasing kills cache performance) or a pre-allocated buffer with a length counter (requires guessing maximum sizes). The KV cache alone — which grows token-by-token during inference — requires a resizable 3D structure (layer x position x embedding_dim) of Value pointers.

## 3. Nested Data Structures Require Manual Offset Arithmetic

The `state_dict` is a dictionary mapping string keys to 2D arrays of Value objects:

```javascript
state_dict = {
  wte: matrix(vocab_size, n_embd),      // 80 x 16 = 1280 Values
  wpe: matrix(block_size, n_embd),      // 16 x 16 = 256 Values
  lm_head: matrix(vocab_size, n_embd),  // 80 x 16 = 1280 Values
  'layer0.attn_wq': matrix(16, 16),     // 256 Values
  // ... 5 more per layer
}
```

In Forth, there are no dictionaries (in the Python sense), no 2D arrays, and no named fields. You must:

1. Decide on a memory layout (e.g., a flat buffer of pointers).
2. Compute the byte offset for `state_dict["layer0.attn_wk"][row][col]` manually.
3. Write access words like `: WTE-AT ( row col -- addr ) SWAP N_EMBD * + CELLS WTE-BASE + ;`
4. Repeat this for every matrix in the state dict.

This is hundreds of lines of boilerplate offset arithmetic that does not exist in the source language. It is not hard — it is tedious, error-prone, and must be done before any actual model code can be written.

## 4. Floating-Point Support Is Awkward

Forth has a **separate** floating-point stack (per ANS Forth). This means:

- You cannot freely mix integer and float operations. Every interaction between an index (integer stack) and a weight (float stack) requires explicit stack juggling.
- Float stack words (`F+`, `F*`, `FDUP`, `FSWAP`, `FROT`) mirror integer stack words but operate on a different stack. You must mentally track two stacks simultaneously.
- Many Forth systems limit the float stack to 6-8 deep. Operations like `softmax` (which needs to compute a max, subtract it from every element, exponentiate, sum, and divide) can easily exceed this.
- `F**` (power), `FEXP`, `FLOG` exist in ANS Forth's optional floating-point extension, but not all implementations include them. You may need to implement `pow` yourself.

The `Value` class compounds this: every "float" is actually a Value node on the heap, not a number on the float stack. So you are constantly dereferencing heap pointers to get floats, operating on them, then storing results back into newly allocated heap nodes. The float stack becomes a scratch area, not a data structure.

## 5. Higher-Order Array Operations Do Not Exist

The codebase leans heavily on `map`, `filter`, and functional patterns:

```javascript
// linear layer
return w.map(wo => vsum(wo.map((wi, i) => wi.mul(x[i]))));

// softmax
const exps = logits.map(v => v.sub(maxVal).exp());

// rmsnorm
const ms = vsum(x.map(xi => xi.mul(xi))).div(x.length);
```

Each of these is a loop over a dynamic array, applying an operation that creates new Value nodes and returns a new dynamic array of results.

In Forth, every `map` becomes a manual `DO ... LOOP` that:
1. Reads a source pointer and length from somewhere.
2. Iterates, dereferencing each element (a Value pointer).
3. Performs the operation (which itself allocates new Value nodes).
4. Stores the result pointer into a destination buffer.
5. Returns the destination buffer address and its length on the stack.

A single line like `w.map(wo => vsum(wo.map(...)))` — a nested map with an inner reduction — becomes 15-25 lines of Forth with manual index management, buffer allocation, and stack manipulation. The `linear` function alone (5 lines in JS) would be 40-60 lines in Forth.

## 6. The Backward Pass Requires Topological Sort

```javascript
backward() {
  const topo = [];
  const visited = new Set();
  function buildTopo(v) {
    if (!visited.has(v)) {
      visited.add(v);
      for (const child of v._children) buildTopo(child);
      topo.push(v);
    }
  }
  buildTopo(this);
  // ...
}
```

This requires:

- **A Set** (for visited tracking): Forth has no hash set. You must implement one (hash table with chaining/probing) or use a linear scan over an array of seen pointers — O(n^2) for the thousands of nodes in the graph.
- **Recursion**: Forth supports recursion via `RECURSE`, but the return stack is shallow (often 32-128 entries). A computation graph with thousands of nodes will overflow it. You need to convert to an explicit iterative traversal with a manual stack — allocated where? On the heap, which you are already manually managing.
- **A dynamic list** (`topo`): Another manually managed resizable buffer.

The backward pass — 15 lines in JS — becomes a major subsystem in Forth: hash set implementation + iterative DFS with manual stack + dynamic result buffer.

## 7. No Closures or Lexical Scope

The `gpt` function closes over `state_dict`, `n_layer`, `n_head`, `head_dim`, and the helper functions `linear`, `softmax`, `rmsnorm`. In JS, this is implicit.

Forth has a single global namespace of words. There are no closures, no modules, and no lexical scope. All state must be communicated through:

- The stack (which is already overloaded with array pointers, indices, and loop counters).
- Global `VARIABLE`s (which makes the code non-reentrant and harder to reason about).
- Passing extra arguments explicitly to every word.

The `gpt` function takes 4 arguments in JS. In Forth, it would effectively need 10+ values available (model dimensions, all weight matrix base addresses, the KV cache pointers), most of which would be globals.

## 8. String-to-Token Mapping

```javascript
const uchars = [...new Set(docs.join(''))].sort();
// tokenize: uchars.indexOf(ch)
// detokenize: uchars[token_id]
```

Building a sorted unique character set from the entire dataset requires:

- Reading the full file into memory.
- Iterating every character and inserting into a set (no native set in Forth).
- Sorting the result (no native sort in Forth — implement quicksort/mergesort yourself).
- `indexOf` for encoding requires linear scan per character (or implement a lookup table).

This preprocessing step, which is one line in JS, is a substantial Forth subprogram.

## 9. The Scale of Manual Labor

Here is a rough accounting of what Forth demands that JS provides for free:

| Subsystem | JS Lines | Estimated Forth Lines | Notes |
|-----------|----------|----------------------|-------|
| Memory allocator / arena | 0 | 60-100 | Pool allocator for Value nodes |
| Value struct + operations | 50 | 120-180 | Manual field access, 8 arithmetic words |
| Dynamic array library | 0 | 60-80 | Push, index, length, allocate |
| Hash set (for backward) | 0 | 40-60 | Needed for visited tracking |
| Topological sort (iterative) | 15 | 50-70 | Manual stack to avoid return stack overflow |
| State dict layout + accessors | 5 | 60-80 | Offset computation for every matrix |
| linear / softmax / rmsnorm | 15 | 80-120 | Nested loops, buffer management |
| gpt function | 40 | 150-200 | All of the above combined |
| Training loop + Adam | 30 | 80-100 | Relatively straightforward |
| Tokenizer + file I/O | 10 | 60-80 | Set construction, sorting, file words |
| **Total** | **~165** | **~760-1070** | **4-6x expansion** |

The expansion factor is not due to Forth being verbose — Forth words are famously terse. It is because Forth provides almost none of the infrastructure this algorithm relies on. You are not porting a neural network. You are first writing a memory manager, a data structure library, and a graph algorithm library, and *then* porting a neural network.

## 10. Debugging Is Brutal

When (not if) something goes wrong:

- There is no stack trace. A segfault from a bad pointer dereference gives you nothing.
- There is no type system. A Value pointer and an integer index are both cells on the stack. Confuse them and you corrupt memory silently.
- There is no `console.log(obj)` — printing a Value node means manually fetching and printing each field.
- Off-by-one errors in manual offset arithmetic are invisible until they produce wrong gradients, which manifest as the loss not decreasing, which could have a hundred other causes.

The debugging cycle for autograd is already painful in high-level languages (gradient checking, numerical stability issues). In Forth, you add pointer corruption, stack underflow, and silent memory overwrites to the mix.

## Conclusion

The difficulty is not that Forth *cannot* express this algorithm — it is Turing complete, and people have written remarkable systems in Forth. The difficulty is the **impedance mismatch**: microgpt is built on heap-allocated object graphs, dynamic arrays, higher-order functions, and automatic memory management. These are the exact things Forth deliberately does not provide, because Forth was designed for direct hardware control with minimal abstraction.

Porting microgpt to Forth means building a substantial runtime system (allocator, data structures, graph algorithms) before writing any machine learning code. The neural network itself is the easy part. The infrastructure to support it is where all the time goes.
