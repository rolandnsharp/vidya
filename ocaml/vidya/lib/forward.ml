(* forward.ml — GPU-backed training + inference forward passes
   =============================================================

   Two forward pass variants:
   1. gpt_forward_batch: training — all S positions at once via
      batched GPU matmuls. Builds autograd graph for backward.
   2. gpt_forward: inference — single token with KV cache.

   All tensor operations dispatch to GPU kernels. The OCaml side
   orchestrates the computation graph; CUDA does the math.

   Blog note: this is where Schuurmans' insight about compute
   budgets meets our architecture. Each attention layer gets
   full GPU-parallel compute — no constant-budget bottleneck.
   The variable compute comes from the number of layers traversed
   and the dictionary lookup that happens outside the transformer. *)

(* ── RoPE tables on GPU ──────────────────────────────────────────── *)

(* Upload the precomputed cos/sin RoPE tables to GPU once at init.
   Stored as flat [block_size, half_dim] float32 buffers. *)
let rope_cos_gpu = lazy (
  let n = Model.block_size * Model.half_dim in
  let flat = Array.create_float n in
  for pos = 0 to Model.block_size - 1 do
    for i = 0 to Model.half_dim - 1 do
      flat.(pos * Model.half_dim + i) <- Model.rope_cos.(pos).(i)
    done
  done;
  let buf = Gpu.create n in
  Gpu.upload flat buf;
  buf)

let rope_sin_gpu = lazy (
  let n = Model.block_size * Model.half_dim in
  let flat = Array.create_float n in
  for pos = 0 to Model.block_size - 1 do
    for i = 0 to Model.half_dim - 1 do
      flat.(pos * Model.half_dim + i) <- Model.rope_sin.(pos).(i)
    done
  done;
  let buf = Gpu.create n in
  Gpu.upload flat buf;
  buf)

(* ── Batched training forward ─────────────────────────────────────── *)

(* Batched multi-head attention with RoPE and causal masking.
   All operations run on GPU via cuBLAS and custom CUDA kernels.

   For each head h:
     1. Extract Q_h, K_h, V_h slices from the interleaved layout
     2. Apply RoPE rotation to Q_h and K_h
     3. Compute scores = Q_h @ K_h^T, scale + causal mask
     4. Softmax the scores
     5. Compute output = softmax(scores) @ V_h
     6. Insert output back into interleaved layout

   The head extraction/insertion is done via GPU copy kernels.
   RoPE is applied in-place on the full [S, n_embd] tensor before
   head extraction, avoiding per-head rotation. *)
let batch_fused_attention q k v seq_len =
  let open Model in
  let cos_tab = Lazy.force rope_cos_gpu in
  let sin_tab = Lazy.force rope_sin_gpu in

  (* Copy Q and K data for rotation (don't modify originals yet —
     backward needs the pre-rotation values). *)
  let q_rot = Gpu.create (seq_len * n_embd) in
  Gpu.copy q.Tensor.data q_rot (seq_len * n_embd);
  let k_rot = Gpu.create (seq_len * n_embd) in
  Gpu.copy k.Tensor.data k_rot (seq_len * n_embd);

  (* Apply RoPE to Q and K in-place on GPU. *)
  Gpu.rope_forward q_rot cos_tab sin_tab seq_len n_embd n_head head_dim;
  Gpu.rope_forward k_rot cos_tab sin_tab seq_len n_embd n_head head_dim;

  let scale = 1.0 /. sqrt (float_of_int head_dim) in

  (* Allocate output and per-head weight storage. *)
  let out_data = Gpu.create (seq_len * n_embd) in
  let all_weights = Array.init n_head (fun _ ->
    Gpu.create (seq_len * seq_len)) in

  (* Per-head temporary buffers on GPU. *)
  let q_h = Gpu.create (seq_len * head_dim) in
  let k_h = Gpu.create (seq_len * head_dim) in
  let v_h = Gpu.create (seq_len * head_dim) in
  let scores = Gpu.create (seq_len * seq_len) in
  let attn_h = Gpu.create (seq_len * head_dim) in

  for h = 0 to n_head - 1 do
    let wt = all_weights.(h) in
    (* Extract heads entirely on GPU — no CPU round trips. *)
    Gpu.extract_head q_rot q_h h seq_len n_embd head_dim;
    Gpu.extract_head k_rot k_h h seq_len n_embd head_dim;
    Gpu.extract_head v.Tensor.data v_h h seq_len n_embd head_dim;

    (* Scores[S,S] = Q_h[S,d] @ K_h[S,d]^T — op=2 NT overwrite *)
    Gpu.sgemm 2 seq_len seq_len head_dim q_h k_h scores;

    (* Scale + causal mask on GPU: upper triangle → -inf, lower × scale *)
    Gpu.causal_mask_scale scores scale seq_len;

    (* Softmax: download scores, compute on CPU, re-upload.
       One round trip per head — much better than per-position.
       TODO: GPU batch softmax kernel to eliminate this too. *)
    let wt_cpu = Gpu.download scores in
    for i = 0 to seq_len - 1 do
      let off = i * seq_len in
      let max_s = ref neg_infinity in
      for j = 0 to i do
        if wt_cpu.(off + j) > !max_s then max_s := wt_cpu.(off + j)
      done;
      let sum_exp = ref 0.0 in
      for j = 0 to i do
        let e = exp (wt_cpu.(off + j) -. !max_s) in
        wt_cpu.(off + j) <- e;
        sum_exp := !sum_exp +. e
      done;
      let inv = 1.0 /. !sum_exp in
      for j = 0 to i do
        wt_cpu.(off + j) <- wt_cpu.(off + j) *. inv
      done;
      for j = i + 1 to seq_len - 1 do wt_cpu.(off + j) <- 0.0 done
    done;
    Gpu.upload wt_cpu wt;

    (* Attn_h[S,d] = Weights[S,S] @ V_h[S,d] — op=0 NN overwrite *)
    Gpu.sgemm 0 seq_len head_dim seq_len wt v_h attn_h;

    (* Insert head back into output — entirely on GPU *)
    Gpu.insert_head attn_h out_data h seq_len n_embd head_dim
  done;

  let out_grad = Gpu.create (seq_len * n_embd) in
  let backward () =
    (* Backward pass for attention.
       This mirrors the forward but propagates gradients.
       TEMPORARY: CPU-side backward until GPU kernels are complete. *)
    (* GPU buffers for backward *)
    let dq_rot = Gpu.create (seq_len * n_embd) in
    let dk_rot = Gpu.create (seq_len * n_embd) in

    let dout_h = Gpu.create (seq_len * head_dim) in
    let v_h_bw = Gpu.create (seq_len * head_dim) in
    let dw = Gpu.create (seq_len * seq_len) in
    let dv_h = Gpu.create (seq_len * head_dim) in
    let k_h_bw = Gpu.create (seq_len * head_dim) in
    let q_h_bw = Gpu.create (seq_len * head_dim) in
    let dq_h = Gpu.create (seq_len * head_dim) in
    let dk_h = Gpu.create (seq_len * head_dim) in

    for h = 0 to n_head - 1 do
      let wt = all_weights.(h) in

      Gpu.extract_head out_grad dout_h h seq_len n_embd head_dim;
      Gpu.extract_head v.Tensor.data v_h_bw h seq_len n_embd head_dim;

      (* dWeights = dAttn_h @ V_h^T — op=2 NT overwrite *)
      Gpu.sgemm 2 seq_len seq_len head_dim dout_h v_h_bw dw;

      (* dV_h = Weights^T @ dAttn_h — op=4 TN overwrite *)
      Gpu.sgemm 4 seq_len head_dim seq_len wt dout_h dv_h;
      Gpu.insert_head_accum dv_h v.Tensor.grad h seq_len n_embd head_dim;

      (* Softmax backward — CPU round trip per head *)
      let dw_cpu = Gpu.download dw in
      let wt_cpu = Gpu.download wt in
      for i = 0 to seq_len - 1 do
        let dot = ref 0.0 in
        for j = 0 to i do
          dot := !dot +. dw_cpu.(i * seq_len + j) *. wt_cpu.(i * seq_len + j)
        done;
        for j = 0 to i do
          dw_cpu.(i * seq_len + j) <-
            wt_cpu.(i * seq_len + j) *. (dw_cpu.(i * seq_len + j) -. !dot) *. scale
        done;
        for j = i + 1 to seq_len - 1 do dw_cpu.(i * seq_len + j) <- 0.0 done
      done;
      Gpu.upload dw_cpu dw;

      (* dQ_rot_h = dScores @ K_rot_h — op=0 NN overwrite *)
      Gpu.extract_head k_rot k_h_bw h seq_len n_embd head_dim;
      Gpu.sgemm 0 seq_len head_dim seq_len dw k_h_bw dq_h;
      Gpu.insert_head_accum dq_h dq_rot h seq_len n_embd head_dim;

      (* dK_rot_h = dScores^T @ Q_rot_h — op=4 TN overwrite *)
      Gpu.extract_head q_rot q_h_bw h seq_len n_embd head_dim;
      Gpu.sgemm 4 seq_len head_dim seq_len dw q_h_bw dk_h;
      Gpu.insert_head_accum dk_h dk_rot h seq_len n_embd head_dim
    done;

    (* Inverse RoPE on gradients — on GPU *)
    Gpu.rope_backward dq_rot cos_tab sin_tab seq_len n_embd n_head head_dim;
    Gpu.rope_backward dk_rot cos_tab sin_tab seq_len n_embd n_head head_dim;

    (* Accumulate into q and k gradients *)
    Gpu.add_forward q.Tensor.grad dq_rot q.Tensor.grad (seq_len * n_embd);
    Gpu.add_forward k.Tensor.grad dk_rot k.Tensor.grad (seq_len * n_embd)
  in
  { Tensor.id = Tensor.fresh_id (); data = out_data; grad = out_grad;
    shape = [|seq_len; n_embd|]; children = [|q; k; v|];
    backward_fn = backward }

(* ── Batched transformer block ────────────────────────────────────── *)

(* RMSNorm → Attention → Residual → RMSNorm → MLP → Residual.
   Named functions for each sub-computation improve readability. *)
let apply_attention x layer seq_len =
  let n = Model.n_embd in
  let xn = Tensor.batch_rmsnorm_affine x layer.Model.ln1 seq_len n in
  let q = Tensor.batch_matmul layer.Model.attn_wq xn seq_len n n in
  let k = Tensor.batch_matmul layer.Model.attn_wk xn seq_len n n in
  let v = Tensor.batch_matmul layer.Model.attn_wv xn seq_len n n in
  let attn_out = batch_fused_attention q k v seq_len in
  let projected = Tensor.batch_matmul layer.Model.attn_wo attn_out seq_len n n in
  Tensor.batch_dropout projected seq_len n

let apply_mlp x layer seq_len =
  let n = Model.n_embd in
  let ffn = 4 * n in
  let xn = Tensor.batch_rmsnorm_affine x layer.Model.ln2 seq_len n in
  let hidden = Tensor.batch_matmul layer.Model.mlp_fc1 xn seq_len ffn n |> Tensor.gelu in
  let out = Tensor.batch_matmul layer.Model.mlp_fc2 hidden seq_len n ffn in
  Tensor.batch_dropout out seq_len n

let batch_transformer_block x layer seq_len =
  let after_attn = Tensor.add x (apply_attention x layer seq_len) in
  Tensor.add after_attn (apply_mlp after_attn layer seq_len)

(* ── Full training forward pass ───────────────────────────────────── *)

(* Returns logits [S, vocab_size] with full autograd graph. *)
let gpt_forward_batch model tokens seq_len =
  let n = Model.n_embd in
  let x = Tensor.batch_embed model.Model.wte tokens n in
  let x = Tensor.batch_rmsnorm_affine x model.Model.embed_norm seq_len n in
  let x = Tensor.batch_dropout x seq_len n in
  let x = Array.fold_left (fun x layer ->
    batch_transformer_block x layer seq_len
  ) x model.Model.layers in
  let x = Tensor.batch_rmsnorm_affine x model.Model.final_norm seq_len n in
  let vocab = model.Model.wte.Tensor.shape.(0) in
  Tensor.batch_matmul model.Model.lm_head x seq_len vocab n

(* ── Inference forward (single token, KV cache) ──────────────────── *)

(* Embed a single token and normalise. *)
let embed_token model token_id =
  Tensor.row model.Model.wte token_id
  |> fun x -> Tensor.rmsnorm_affine x model.Model.embed_norm Model.n_embd

(* Single-token attention with KV cache.
   TEMPORARY: CPU-side implementation until GPU inference path is built.
   Training is the priority; inference comes after. *)
let fused_multi_head_attention x layer (kv : Model.kv_cache) =
  let open Model in
  let q_raw = Tensor.matmul layer.attn_wq x n_embd n_embd in
  let k_raw = Tensor.matmul layer.attn_wk x n_embd n_embd in
  let v = Tensor.matmul layer.attn_wv x n_embd n_embd in
  let pos = kv.len in

  (* Download to CPU for KV cache management and attention.
     This is the inference path — not performance-critical yet. *)
  let q_cpu = Tensor.to_cpu q_raw.data in
  let k_cpu = Tensor.to_cpu k_raw.data in
  let v_cpu = Tensor.to_cpu v.data in

  let q_rot = Array.create_float n_embd in
  let cos_p = rope_cos.(pos) and sin_p = rope_sin.(pos) in
  for h = 0 to n_head - 1 do
    let h_off = h * head_dim in
    for i = 0 to half_dim - 1 do
      let j0 = h_off + 2 * i and j1 = h_off + 2 * i + 1 in
      let c = cos_p.(i) and s = sin_p.(i) in
      q_rot.(j0) <- q_cpu.(j0) *. c -. q_cpu.(j1) *. s;
      q_rot.(j1) <- q_cpu.(j0) *. s +. q_cpu.(j1) *. c;
      kv.k_cache.(h).(pos * head_dim + 2 * i) <-
        k_cpu.(j0) *. c -. k_cpu.(j1) *. s;
      kv.k_cache.(h).(pos * head_dim + 2 * i + 1) <-
        k_cpu.(j0) *. s +. k_cpu.(j1) *. c
    done;
    for j = 0 to head_dim - 1 do
      kv.v_cache.(h).(pos * head_dim + j) <- v_cpu.(h_off + j)
    done
  done;
  kv.len <- pos + 1;
  let t_len = pos + 1 in
  let scale = 1.0 /. sqrt (float_of_int head_dim) in
  let out_data = Array.create_float n_embd in
  for h = 0 to n_head - 1 do
    let h_off = h * head_dim in
    let weights = Array.create_float t_len in
    let max_score = ref neg_infinity in
    for t = 0 to t_len - 1 do
      let s = ref 0.0 in
      for j = 0 to head_dim - 1 do
        s := !s +. kv.k_cache.(h).(t * head_dim + j) *. q_rot.(h_off + j)
      done;
      let score = !s *. scale in
      weights.(t) <- score;
      if score > !max_score then max_score := score
    done;
    let sum_exp = ref 0.0 in
    for t = 0 to t_len - 1 do
      let e = exp (weights.(t) -. !max_score) in
      weights.(t) <- e;
      sum_exp := !sum_exp +. e
    done;
    let inv_sum = 1.0 /. !sum_exp in
    for t = 0 to t_len - 1 do weights.(t) <- weights.(t) *. inv_sum done;
    for j = 0 to head_dim - 1 do
      let s = ref 0.0 in
      for t = 0 to t_len - 1 do
        s := !s +. weights.(t) *. kv.v_cache.(h).(t * head_dim + j)
      done;
      out_data.(h_off + j) <- !s
    done
  done;
  (* Upload result back to GPU. *)
  let out_gpu = Gpu.create n_embd in
  Gpu.upload out_data out_gpu;
  let out_grad = Gpu.create n_embd in
  let attn_out = { Tensor.id = Tensor.fresh_id ();
    data = out_gpu; grad = out_grad;
    shape = [|n_embd|]; children = [||];
    backward_fn = (fun () -> ()) } in
  Tensor.matmul layer.attn_wo attn_out n_embd n_embd

let mlp_block x layer =
  let n = Model.n_embd in
  Tensor.matmul layer.Model.mlp_fc1 x (4 * n) n
  |> Tensor.gelu
  |> fun h -> Tensor.matmul layer.Model.mlp_fc2 h n (4 * n)

let transformer_block x layer kv =
  let n = Model.n_embd in
  let x =
    Tensor.rmsnorm_affine x layer.Model.ln1 n |> fun xn ->
    fused_multi_head_attention xn layer kv |> Tensor.add x in
  Tensor.rmsnorm_affine x layer.Model.ln2 n |> fun xn ->
    mlp_block xn layer |> Tensor.add x

(* Single-token inference. Returns logits [vocab_size]. *)
let gpt_forward model token_id kv_caches =
  let x = embed_token model token_id in
  let x = Array.to_list model.Model.layers
    |> List.mapi (fun li layer -> (li, layer))
    |> List.fold_left (fun x (li, layer) ->
      transformer_block x layer kv_caches.(li)) x in
  let x = Tensor.rmsnorm_affine x model.Model.final_norm Model.n_embd in
  let vocab = model.Model.wte.Tensor.shape.(0) in
  Tensor.matmul model.Model.lm_head x vocab Model.n_embd
