(* forward.ml — Training + inference forward passes
   ==================================================

   Two forward pass variants:
   1. gpt_forward_batch: training — processes all S positions at once
      via batched matmuls. Builds the full autograd graph for backward.
   2. gpt_forward: inference — single token at a time with KV cache.
      No autograd needed (backward is a no-op). *)

(* ── Batched training forward ─────────────────────────────────────── *)

(* Batched multi-head attention with RoPE, causal mask, and BLAS.

   BLAS ops per head (forward):
     Scores[S,S] = Q_h[S,d] @ K_h[S,d]^T     op=2 (NT overwrite)
     Attn_h[S,d] = Weights[S,S] @ V_h[S,d]    op=0 (NN overwrite)

   BLAS ops per head (backward):
     dWeights = dAttn_h @ V_h^T              op=2
     dV_h     = Weights^T @ dAttn_h          op=4 (TN overwrite)
     dQ_rot_h = dScores @ K_rot_h            op=0
     dK_rot_h = dScores^T @ Q_rot_h          op=4

   Scalar loops handle: causal mask, softmax fwd/bwd, RoPE rotation. *)
let batch_fused_attention q k v seq_len =
  let n_embd = Model.n_embd in
  let n_head = Model.n_head in
  let head_dim = Model.head_dim in
  let half_dim = Model.half_dim in

  (* Apply RoPE rotation to all positions of Q and K *)
  let q_rot = Array.create_float (seq_len * n_embd) in
  let k_rot = Array.create_float (seq_len * n_embd) in
  for pos = 0 to seq_len - 1 do
    let cos_p = Model.rope_cos.(pos) and sin_p = Model.rope_sin.(pos) in
    let row = pos * n_embd in
    for h = 0 to n_head - 1 do
      let base = row + h * head_dim in
      for i = 0 to half_dim - 1 do
        let j0 = base + 2 * i and j1 = base + 2 * i + 1 in
        let c = cos_p.(i) and s = sin_p.(i) in
        q_rot.(j0) <- q.Tensor.data.(j0) *. c -. q.Tensor.data.(j1) *. s;
        q_rot.(j1) <- q.Tensor.data.(j0) *. s +. q.Tensor.data.(j1) *. c;
        k_rot.(j0) <- k.Tensor.data.(j0) *. c -. k.Tensor.data.(j1) *. s;
        k_rot.(j1) <- k.Tensor.data.(j0) *. s +. k.Tensor.data.(j1) *. c
      done
    done
  done;

  let scale = 1.0 /. sqrt (float_of_int head_dim) in
  let out_data = Array.create_float (seq_len * n_embd) in
  let all_weights = Array.init n_head (fun _ ->
    Array.create_float (seq_len * seq_len)) in

  (* Temporary buffers for head extraction — reused across heads *)
  let q_h = Array.create_float (seq_len * head_dim) in
  let k_h = Array.create_float (seq_len * head_dim) in
  let v_h = Array.create_float (seq_len * head_dim) in
  let scores = Array.create_float (seq_len * seq_len) in
  let attn_h = Array.create_float (seq_len * head_dim) in

  for h = 0 to n_head - 1 do
    let wt = all_weights.(h) in

    Model.extract_head q_rot q_h h seq_len;
    Model.extract_head k_rot k_h h seq_len;

    (* Scores[S,S] = Q_h[S,d] @ K_h[S,d]^T *)
    Blas.dgemm 2 seq_len seq_len head_dim q_h k_h scores;

    (* Scale + causal mask + row-wise softmax *)
    for i = 0 to seq_len - 1 do
      let max_s = ref neg_infinity in
      for j = 0 to i do
        let s = scores.(i * seq_len + j) *. scale in
        wt.(i * seq_len + j) <- s;
        if s > !max_s then max_s := s
      done;
      let sum_exp = ref 0.0 in
      for j = 0 to i do
        let e = exp (wt.(i * seq_len + j) -. !max_s) in
        wt.(i * seq_len + j) <- e;
        sum_exp := !sum_exp +. e
      done;
      let inv = 1.0 /. !sum_exp in
      for j = 0 to i do
        wt.(i * seq_len + j) <- wt.(i * seq_len + j) *. inv
      done;
      for j = i + 1 to seq_len - 1 do wt.(i * seq_len + j) <- 0.0 done
    done;

    (* Attn_h[S,d] = Weights[S,S] @ V_h[S,d] *)
    Model.extract_head v.Tensor.data v_h h seq_len;
    Blas.dgemm 0 seq_len head_dim seq_len wt v_h attn_h;

    Model.insert_head attn_h out_data h seq_len
  done;

  let out_grad = Array.make (seq_len * n_embd) 0.0 in
  let backward () =
    let dq_rot = Array.make (seq_len * n_embd) 0.0 in
    let dk_rot = Array.make (seq_len * n_embd) 0.0 in

    let dout_h = Array.create_float (seq_len * head_dim) in
    let v_h = Array.create_float (seq_len * head_dim) in
    let dw = Array.create_float (seq_len * seq_len) in
    let dv_h = Array.create_float (seq_len * head_dim) in
    let k_h = Array.create_float (seq_len * head_dim) in
    let q_h = Array.create_float (seq_len * head_dim) in
    let dq_h = Array.create_float (seq_len * head_dim) in
    let dk_h = Array.create_float (seq_len * head_dim) in

    for h = 0 to n_head - 1 do
      let wt = all_weights.(h) in

      Model.extract_head out_grad dout_h h seq_len;
      Model.extract_head v.Tensor.data v_h h seq_len;

      (* dWeights = dAttn_h @ V_h^T *)
      Blas.dgemm 2 seq_len seq_len head_dim dout_h v_h dw;

      (* dV_h = Weights^T @ dAttn_h *)
      Blas.dgemm 4 seq_len head_dim seq_len wt dout_h dv_h;
      Model.insert_head_accum dv_h v.Tensor.grad h seq_len;

      (* Softmax backward *)
      for i = 0 to seq_len - 1 do
        let dot = ref 0.0 in
        for j = 0 to i do
          dot := !dot +. dw.(i * seq_len + j) *. wt.(i * seq_len + j)
        done;
        for j = 0 to i do
          dw.(i * seq_len + j) <-
            wt.(i * seq_len + j) *. (dw.(i * seq_len + j) -. !dot) *. scale
        done;
        for j = i + 1 to seq_len - 1 do dw.(i * seq_len + j) <- 0.0 done
      done;

      (* dQ_rot_h = dScores @ K_rot_h *)
      Model.extract_head k_rot k_h h seq_len;
      Blas.dgemm 0 seq_len head_dim seq_len dw k_h dq_h;
      Model.insert_head_accum dq_h dq_rot h seq_len;

      (* dK_rot_h = dScores^T @ Q_rot_h *)
      Model.extract_head q_rot q_h h seq_len;
      Blas.dgemm 4 seq_len head_dim seq_len dw q_h dk_h;
      Model.insert_head_accum dk_h dk_rot h seq_len
    done;

    (* Inverse RoPE: rotate gradients back to pre-rotation space *)
    for pos = 0 to seq_len - 1 do
      let cos_p = Model.rope_cos.(pos) and sin_p = Model.rope_sin.(pos) in
      let row = pos * n_embd in
      for h = 0 to n_head - 1 do
        let base = row + h * head_dim in
        for i = 0 to half_dim - 1 do
          let j0 = base + 2 * i and j1 = base + 2 * i + 1 in
          let c = cos_p.(i) and s = sin_p.(i) in
          q.Tensor.grad.(j0) <- q.Tensor.grad.(j0) +. dq_rot.(j0) *. c +. dq_rot.(j1) *. s;
          q.Tensor.grad.(j1) <- q.Tensor.grad.(j1) -. dq_rot.(j0) *. s +. dq_rot.(j1) *. c;
          k.Tensor.grad.(j0) <- k.Tensor.grad.(j0) +. dk_rot.(j0) *. c +. dk_rot.(j1) *. s;
          k.Tensor.grad.(j1) <- k.Tensor.grad.(j1) -. dk_rot.(j0) *. s +. dk_rot.(j1) *. c
        done
      done
    done
  in
  { Tensor.id = Tensor.fresh_id (); data = out_data; grad = out_grad;
    shape = [|seq_len; n_embd|]; children = [|q; k; v|];
    backward_fn = backward }

(* Batched transformer block: RMSNorm → Attention → Residual → RMSNorm → MLP → Residual *)
let batch_transformer_block x layer seq_len =
  let n_embd = Model.n_embd in
  let m = n_embd and n = n_embd in
  let xn = Tensor.batch_rmsnorm x seq_len n_embd in
  let q = Tensor.batch_matmul layer.Model.attn_wq xn seq_len m n in
  let k = Tensor.batch_matmul layer.Model.attn_wk xn seq_len m n in
  let v = Tensor.batch_matmul layer.Model.attn_wv xn seq_len m n in
  let attn_out = batch_fused_attention q k v seq_len in
  let attn_proj = Tensor.batch_matmul layer.Model.attn_wo attn_out seq_len m n in
  let x = Tensor.add x attn_proj in
  let xn = Tensor.batch_rmsnorm x seq_len n_embd in
  let mlp_h = Tensor.batch_matmul layer.Model.mlp_fc1 xn seq_len (4 * n_embd) n_embd |> Tensor.gelu in
  let mlp_out = Tensor.batch_matmul layer.Model.mlp_fc2 mlp_h seq_len n_embd (4 * n_embd) in
  Tensor.add x mlp_out

(* gpt_forward_batch: Full training forward pass over a token sequence.
   Returns logits [S, vocab_size]. *)
let gpt_forward_batch model tokens seq_len =
  let n_embd = Model.n_embd in
  let x = Tensor.batch_embed model.Model.wte tokens n_embd in
  let x = Tensor.batch_rmsnorm x seq_len n_embd in
  let x = Array.fold_left (fun x layer ->
    batch_transformer_block x layer seq_len
  ) x model.Model.layers in
  let vocab_size = model.Model.wte.Tensor.shape.(0) in
  Tensor.batch_matmul model.Model.lm_head x seq_len vocab_size n_embd

(* ── Inference forward (single token, KV cache) ──────────────────── *)

let embed_token model token_id =
  let n_embd = Model.n_embd in
  Tensor.row model.Model.wte token_id |> fun x -> Tensor.rmsnorm x n_embd

(* Single-token multi-head attention with KV cache.
   Appends current K,V to cache; attends over full history. *)
let fused_multi_head_attention x layer (kv : Model.kv_cache) =
  let n_embd = Model.n_embd in
  let n_head = Model.n_head in
  let head_dim = Model.head_dim in
  let half_dim = Model.half_dim in

  let q_raw = Tensor.matmul layer.Model.attn_wq x n_embd n_embd in
  let k_raw = Tensor.matmul layer.Model.attn_wk x n_embd n_embd in
  let v = Tensor.matmul layer.Model.attn_wv x n_embd n_embd in
  let pos = kv.len in
  kv.k_nodes.(pos) <- k_raw;
  kv.v_nodes.(pos) <- v;

  let q_rot = Array.create_float n_embd in
  let cos_p = Model.rope_cos.(pos) and sin_p = Model.rope_sin.(pos) in
  for h = 0 to n_head - 1 do
    let h_off = h * head_dim in
    for i = 0 to half_dim - 1 do
      let j0 = h_off + 2 * i and j1 = h_off + 2 * i + 1 in
      let c = cos_p.(i) and s = sin_p.(i) in
      q_rot.(j0) <- q_raw.data.(j0) *. c -. q_raw.data.(j1) *. s;
      q_rot.(j1) <- q_raw.data.(j0) *. s +. q_raw.data.(j1) *. c;
      kv.k_cache.(h).(pos * head_dim + 2 * i) <-
        k_raw.data.(j0) *. c -. k_raw.data.(j1) *. s;
      kv.k_cache.(h).(pos * head_dim + 2 * i + 1) <-
        k_raw.data.(j0) *. s +. k_raw.data.(j1) *. c
    done;
    for j = 0 to head_dim - 1 do
      kv.v_cache.(h).(pos * head_dim + j) <- v.data.(h_off + j)
    done
  done;
  kv.len <- pos + 1;
  let t_len = pos + 1 in
  let scale = 1.0 /. sqrt (float_of_int head_dim) in
  let out_data = Array.create_float n_embd in
  let all_weights = Array.init n_head (fun _ -> Array.create_float t_len) in
  for h = 0 to n_head - 1 do
    let h_off = h * head_dim in
    let weights = all_weights.(h) in
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
    for t = 0 to t_len - 1 do
      weights.(t) <- weights.(t) *. inv_sum
    done;
    for j = 0 to head_dim - 1 do
      let s = ref 0.0 in
      for t = 0 to t_len - 1 do
        s := !s +. weights.(t) *. kv.v_cache.(h).(t * head_dim + j)
      done;
      out_data.(h_off + j) <- !s
    done
  done;
  let out_grad = Array.make n_embd 0.0 in
  let backward () = () in
  let attn_out = { Tensor.id = Tensor.fresh_id (); data = out_data; grad = out_grad;
    shape = [|n_embd|]; children = [||]; backward_fn = backward } in
  Tensor.matmul layer.Model.attn_wo attn_out n_embd n_embd

let mlp_block x layer =
  let n_embd = Model.n_embd in
  Tensor.matmul layer.Model.mlp_fc1 x (4 * n_embd) n_embd
  |> Tensor.gelu
  |> fun h -> Tensor.matmul layer.Model.mlp_fc2 h n_embd (4 * n_embd)

let transformer_block x layer kv =
  let n_embd = Model.n_embd in
  let x =
    Tensor.rmsnorm x n_embd |> fun xn ->
    fused_multi_head_attention xn layer kv |> Tensor.add x in
  Tensor.rmsnorm x n_embd |> fun xn -> mlp_block xn layer |> Tensor.add x

(* gpt_forward: Single-token inference with KV cache.
   Returns logits [vocab_size]. *)
let gpt_forward model token_id kv_caches =
  let n_embd = Model.n_embd in
  let x = embed_token model token_id in
  let x = Array.to_list model.Model.layers
    |> List.mapi (fun li layer -> (li, layer))
    |> List.fold_left (fun x (li, layer) ->
      transformer_block x layer kv_caches.(li)) x in
  let vocab_size = model.Model.wte.Tensor.shape.(0) in
  Tensor.matmul model.Model.lm_head x vocab_size n_embd
