# NimLLM TODO

## Performance
- [ ] Optimize flash attention kernel with shared memory tiling (close 2-5x gap with PyTorch)
- [ ] Fused flash attention backward with tiling (currently naive O(S²×hd) per thread)
- [ ] Pre-allocate scratch buffers instead of per-step trackedCreate/free
- [ ] BPE encode optimization: hash table lookup instead of iterating all merge rules

## Features
- [ ] `nimllm chat` — interactive conversation mode with retraining
- [ ] `nimllm read` — absorb a document into weights
- [ ] `nimllm code` — agent mode via Girvent, executes shell commands
- [ ] Checkpoint save/load for microgpt model
- [ ] Progressive model growing: expand dims/layers while preserving learned weights
- [ ] Claude Code JSONL log ingestion for training

## Architecture
- [ ] SwiGLU activation (kernel ready in kernels.cu, not wired into microgpt)
- [ ] GQA — grouped query attention (kernel ready, not wired in)
- [ ] Dropout (kernel ready, not wired in)
- [ ] RoPE instead of positional embeddings
- [ ] Weight tying (lm_head = wte)
- [ ] Larger vocab (8K+ merges, tokenizer_v2.bin already trained)

## Memory Mechanism
- [ ] Sparse gradient masking: top 1% of gradients per interaction
- [ ] Elastic weight consolidation: pull weights toward base model
- [ ] Interactive RL: retrain on chosen/rejected responses
- [ ] Sleep cycle: consolidate day's learning into base weights

## Deployment
- [ ] Simple Nim TUI for training dashboard (loss curve, GPU stats)
- [ ] Safetensors export for HuggingFace publishing
- [ ] Single binary packaging (embed tokenizer in binary)
- [ ] Tenstorrent Blackhole port (Nim → C++ → TT-Metalium)
