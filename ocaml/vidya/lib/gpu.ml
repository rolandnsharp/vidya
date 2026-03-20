(* gpu.ml — GPU tensor type and FFI declarations
   ================================================

   Opaque gpu_buf type backed by cudaMalloc'd float32 device memory.
   OCaml custom blocks with finalizers handle deallocation.
   All numerical work dispatches through these externals to CUDA. *)

type gpu_buf

(* ── Memory management ───────────────────────────────────────────── *)

(* Allocate n float32 elements on GPU, zero-initialized. *)
external create : int -> gpu_buf = "caml_gpu_create"

(* Upload float64 OCaml array → float32 GPU buffer (with conversion). *)
external upload : float array -> gpu_buf -> unit = "caml_gpu_upload"

(* Download float32 GPU buffer → fresh float64 OCaml array. *)
external download : gpu_buf -> float array = "caml_gpu_download"

(* Number of float32 elements in buffer. *)
external numel : gpu_buf -> int = "caml_gpu_numel"

(* Zero all elements. *)
external zero : gpu_buf -> unit = "caml_gpu_zero"

(* ── BLAS ────────────────────────────────────────────────────────── *)

(* sgemm: same 3-bit op encoding as the CPU dgemm.
   bit 2 (4): transpose A
   bit 1 (2): transpose B
   bit 0 (1): accumulate (beta=1) vs overwrite (beta=0) *)
external sgemm : int -> int -> int -> int
  -> gpu_buf -> gpu_buf -> gpu_buf -> unit
  = "caml_gpu_sgemm_byte" "caml_gpu_sgemm"

(* ── Element-wise ops ────────────────────────────────────────────── *)

external gelu_forward : gpu_buf -> gpu_buf -> int -> unit
  = "caml_gpu_gelu_forward"

external gelu_backward : gpu_buf -> gpu_buf -> gpu_buf -> int -> unit
  = "caml_gpu_gelu_backward"

external add_forward : gpu_buf -> gpu_buf -> gpu_buf -> int -> unit
  = "caml_gpu_add_forward"

external scale_forward : gpu_buf -> float -> gpu_buf -> int -> unit
  = "caml_gpu_scale_forward"

external dropout_forward : gpu_buf -> gpu_buf -> gpu_buf -> float -> int -> unit
  = "caml_gpu_dropout_forward"

external dropout_backward : gpu_buf -> gpu_buf -> gpu_buf -> int -> unit
  = "caml_gpu_dropout_backward"

(* ── Reductions ──────────────────────────────────────────────────── *)

external softmax_forward : gpu_buf -> gpu_buf -> int -> unit
  = "caml_gpu_softmax_forward"

external softmax_backward : gpu_buf -> gpu_buf -> gpu_buf -> int -> unit
  = "caml_gpu_softmax_backward"

external rmsnorm_forward : gpu_buf -> gpu_buf -> gpu_buf -> int -> int -> unit
  = "caml_gpu_rmsnorm_forward"

external rmsnorm_affine_forward :
  gpu_buf -> gpu_buf -> gpu_buf -> gpu_buf -> int -> int -> unit
  = "caml_gpu_rmsnorm_affine_forward_byte" "caml_gpu_rmsnorm_affine_forward"

external rmsnorm_backward : gpu_buf -> gpu_buf -> gpu_buf -> gpu_buf -> int -> int -> unit
  = "caml_gpu_rmsnorm_backward_byte" "caml_gpu_rmsnorm_backward"

(* x, gamma, dy, norm_data, rms, dx, dgamma, rows, dim *)
external rmsnorm_affine_backward :
  gpu_buf -> gpu_buf -> gpu_buf -> gpu_buf -> gpu_buf ->
  gpu_buf -> gpu_buf -> int -> int -> unit
  = "caml_gpu_rmsnorm_affine_backward_byte" "caml_gpu_rmsnorm_affine_backward"

(* ── Embedding ───────────────────────────────────────────────────── *)

(* tokens is an int array on CPU; wte is gpu_buf [vocab, n_embd]. *)
external embed_forward : gpu_buf -> int array -> gpu_buf -> int -> int -> unit
  = "caml_gpu_embed_forward"

external embed_backward : gpu_buf -> int array -> gpu_buf -> int -> int -> unit
  = "caml_gpu_embed_backward"

(* ── Loss ────────────────────────────────────────────────────────── *)

(* NLL: -log(probs[target]). Returns scalar float on CPU. *)
external nll_forward : gpu_buf -> int -> float = "caml_gpu_nll_forward"

(* NLL backward: probs_grad[target] -= 1/probs[target] * upstream. *)
external nll_backward : gpu_buf -> gpu_buf -> int -> float -> unit
  = "caml_gpu_nll_backward"

(* ── Row / slice ops ─────────────────────────────────────────────── *)

(* Copy row i from [rows, cols] matrix into [cols] vector. *)
external row_forward : gpu_buf -> gpu_buf -> int -> int -> unit
  = "caml_gpu_row_forward"

external row_backward : gpu_buf -> gpu_buf -> int -> int -> unit
  = "caml_gpu_row_backward"

(* ── Head extraction/insertion ─────────────────────────────────── *)

external extract_head : gpu_buf -> gpu_buf -> int -> int -> int -> int -> unit
  = "caml_gpu_extract_head_byte" "caml_gpu_extract_head"

external insert_head : gpu_buf -> gpu_buf -> int -> int -> int -> int -> unit
  = "caml_gpu_insert_head_byte" "caml_gpu_insert_head"

external insert_head_accum : gpu_buf -> gpu_buf -> int -> int -> int -> int -> unit
  = "caml_gpu_insert_head_accum_byte" "caml_gpu_insert_head_accum"

(* ── RoPE ────────────────────────────────────────────────────────── *)

(* Apply rotary position embedding in-place. *)
external rope_forward :
  gpu_buf -> gpu_buf -> gpu_buf -> int -> int -> int -> int -> unit
  = "caml_gpu_rope_forward_byte" "caml_gpu_rope_forward"

external rope_backward :
  gpu_buf -> gpu_buf -> gpu_buf -> int -> int -> int -> int -> unit
  = "caml_gpu_rope_backward_byte" "caml_gpu_rope_backward"

(* ── Causal mask + scale ─────────────────────────────────────────── *)

external causal_mask_scale : gpu_buf -> float -> int -> unit
  = "caml_gpu_causal_mask_scale"

(* ── Optimizer ───────────────────────────────────────────────────── *)

external adam_step :
  gpu_buf -> gpu_buf -> gpu_buf -> gpu_buf ->
  float -> float -> float -> float -> float -> int -> unit
  = "caml_gpu_adam_step_byte" "caml_gpu_adam_step"

external grad_clip_norm : gpu_buf array -> float -> unit
  = "caml_gpu_grad_clip_norm"

external sparse_grad_mask : gpu_buf array -> float -> int
  = "caml_gpu_sparse_grad_mask"

external elastic_pull : gpu_buf -> gpu_buf -> float -> int -> unit
  = "caml_gpu_elastic_pull"

(* ── Device copy ─────────────────────────────────────────────────── *)

(* Copy n floats from src to dst on device. *)
external copy : gpu_buf -> gpu_buf -> int -> unit = "caml_gpu_copy"

(* ── Init ────────────────────────────────────────────────────────── *)

(* Initialize cuBLAS handle and cuRAND generator. Call once at startup. *)
external init : unit -> unit = "caml_gpu_init"
