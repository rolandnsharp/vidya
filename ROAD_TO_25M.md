# Road to 25M

Scaling vidya from 10M to 25M parameters. Everything that needs to happen
before we start a multi-week training run.

## Current Architecture (10M)

| Param       | Value | Notes                          |
|-------------|-------|--------------------------------|
| n_embd      | 256   | Embedding / model dimension    |
| n_layer     | 12    | Transformer blocks             |
| n_head      | 8     | Attention heads                |
| head_dim    | 32    | n_embd / n_head                |
| block_size  | 256   | Context window (tokens)        |
| FFN hidden  | 1024  | 4 × n_embd                     |
| vocab       | ~2188 | 185 chars + 2000 merges + 3    |
| Params      | ~10M  | Weight-tied (lm_head = wte)    |

## Target Architecture (25M)

| Param       | Value | Notes                          |
|-------------|-------|--------------------------------|
| n_embd      | 384   | +50% width                     |
| n_layer     | 18    | +50% depth                     |
| n_head      | 12    | head_dim stays 32              |
| head_dim    | 32    | Same as 10M                    |
| block_size  | 512   | 2× context window              |
| FFN hidden  | 1536  | 4 × 384                        |
| vocab       | ~4200 | Retrained on larger corpus     |
| Params      | ~25M  | Estimate, verify after impl    |

## Checklist

### 1. Retrain BPE tokenizer on combined corpus
- [ ] Combine chat_input.txt (37K docs) + chat_input_ultrachat.txt (208K docs)
- [ ] Bump n_merges from 2000 to 4000 for better compression
- [ ] Delete cached tokenizer_v3.bin so it retrains
- [ ] Verify new vocab size and chars/token ratio
- **Why**: Current tokenizer only knows words from 37K DailyDialog docs.
  UltraChat has much broader vocabulary. Better tokenizer = fewer tokens
  per doc = faster training and more context per window.
- **Files**: `lib/bpe.ml` (n_merges constant), `bin/main.ml` (input_file path)

### 2. Enable multi-threaded BLAS
- [ ] Change thread count from 1 to 4 in blas_stubs.c
- [ ] Benchmark: time a single training step at 10M with 1 vs 4 threads
- [ ] Benchmark: time a single step at 25M with 1 vs 4 threads
- [ ] Pick optimal thread count
- **Why**: At 256×256, single-thread wins. At 384×384, multi-thread should
  help. At 512×512 (FFN), it definitely helps. Could be 2-3x speedup.
- **Files**: `stubs/blas_stubs.c` (openblas_set_num_threads call)

### 3. Scale model.ml hyperparameters
- [ ] n_embd: 256 → 384
- [ ] n_layer: 12 → 18
- [ ] n_head: 8 → 12
- [ ] block_size: 256 → 512
- [ ] FFN hidden dim follows automatically (4 × n_embd = 1536)
- [ ] head_dim stays 32 (384 / 12 = 32)
- [ ] Update residual_std: 0.08 / sqrt(2 × 18) = 0.01333
- [ ] RoPE tables resize automatically (block_size × half_dim)
- [ ] Verify param count after init
- **Why**: This is the core scale-up. Everything else supports this.
- **Files**: `lib/model.ml` (constants at top, init function)

### 4. Tune training hyperparameters
- [ ] Peak LR: 0.001 → 0.0003 (bigger models need lower LR)
- [ ] Warmup steps: 2000 → 5000 (more params need longer warmup)
- [ ] Decide total training steps based on dataset size
  - 208K docs, ~2-3 passes = 400-600K steps
  - At estimated ~3-5s/step = 14-35 days
- [ ] Gradient clipping stays at 1.0 (should be fine)
- **Files**: `lib/train.ml` (learning_rate, warmup_steps constants)

### 5. Update checkpoint naming
- [ ] Checkpoint file: microgpt_chat_25m_v5.bin
- [ ] Train checkpoint: microgpt_chat_25m_v5_train.bin
- [ ] Checkpoint base for intermediates: microgpt_chat_25m_v5
- [ ] collect_params ordering changes = incompatible with 10M checkpoints
- **Files**: `bin/main.ml` (checkpoint_file, train_checkpoint, checkpoint_base)

### 6. Update training data path
- [ ] Point input_file to combined corpus or ultrachat file
- [ ] Decide: ultrachat only (208K) or combined with original (245K)?
- [ ] Verify load_docs handles the file (1.2GB, may need to check memory)
- **Files**: `bin/main.ml` (input_file constant)

### 7. Memory estimation
- Weights: 25M × 8 bytes = 200MB
- Adam state (m + v): 25M × 2 × 8 = 400MB
- Gradients: 200MB
- Activations (512 tokens × 18 layers): ~500MB-1GB
- **Total estimate**: ~1.5-2GB (comfortable in 32GB RAM)

### 8. Training time estimation
- 10M at 256 context: ~0.8s/step
- 25M at 512 context: estimated 3-5s/step
  - Matrices 2.25× larger (384² vs 256²)
  - 1.5× more layers
  - 2× longer sequences
  - Multi-threaded BLAS should offset some of this
- 400K steps × 4s = 18 days
- 600K steps × 4s = 28 days

### 9. Pre-flight checks
- [ ] dune build succeeds with new params
- [ ] Model init prints correct param count (~25M)
- [ ] Single training step runs without OOM
- [ ] Tokenizer trains on combined corpus, saves correctly
- [ ] Checkpoint saves/loads at new size
- [ ] --resume works with new checkpoint naming

## Order of operations

1. Retrain tokenizer (fast, changes vocab size)
2. Enable multi-threaded BLAS + benchmark
3. Scale model.ml
4. Update train.ml hyperparams
5. Update main.ml (paths, naming)
6. Pre-flight: build, init, single step test
7. Start training run
8. Monitor loss curve, verify it's learning
9. After training: personality fine-tuning + interactive RL

## What we're NOT changing
- Architecture type (pre-norm transformer, RoPE, weight-tied)
- Activation function (GELU)
- Dropout rate (0.1)
- Top-k sampling (40)
- Adam optimizer (beta1=0.9, beta2=0.999)
- Gradient clipping (max norm 1.0)
- No symbolic constraints (staying clean for RL experiments)
