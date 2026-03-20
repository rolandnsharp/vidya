(* train.ml — GPU-backed Adam optimizer, training loop, checkpoints
   ====================================================================

   Adam with cosine LR schedule, warmup, and gradient clipping.
   All optimizer operations run on GPU via per-parameter kernel launches.

   The "frontal cortex" retraining mechanism lives here:
   - sparse_grad_mask: only top 1% of gradients update (the synapses
     that fired hardest). This is our selective retraining.
   - elastic_pull: pulls weights back toward the base model after
     each RL step. Weights that consistently fire hard resist the pull.
   Together these give the model persistent memory without catastrophic
   forgetting — the key experiment we need the GPU to test at 103M. *)

(* ── Adam state ──────────────────────────────────────────────────── *)

(* Per-parameter GPU buffers for Adam first and second moments.
   One pair of buffers per parameter tensor, matching its size. *)
type adam_state = {
  m : Gpu.gpu_buf array;   (* first moments, one per param tensor *)
  v : Gpu.gpu_buf array;   (* second moments, one per param tensor *)
}

let learning_rate = 0.0003   (* lower for 103M model *)
let beta1 = 0.9
let beta2 = 0.999
let max_grad_norm = 1.0
let warmup_steps = 5000      (* longer warmup for larger model *)

(* Allocate zeroed GPU buffers for Adam moments, one per parameter. *)
let init_adam params =
  let m = Array.map (fun p -> Gpu.create (Gpu.numel p.Tensor.data)) params in
  let v = Array.map (fun p -> Gpu.create (Gpu.numel p.Tensor.data)) params in
  { m; v }

(* ── Learning rate schedule ──────────────────────────────────────── *)

(* Cosine annealing with linear warmup. *)
let get_lr step num_steps =
  if step < warmup_steps then
    learning_rate *. float_of_int step /. float_of_int warmup_steps
  else
    let progress =
      float_of_int (step - warmup_steps)
      /. float_of_int (num_steps - warmup_steps) in
    learning_rate *. 0.5 *. (1.0 +. cos (Float.pi *. progress))

(* ── Gradient operations ─────────────────────────────────────────── *)

(* Sparse gradient mask: keep only top fraction of gradients by
   magnitude, zero the rest. This is our "frontal cortex" — only
   the weights most activated by this interaction get updated.

   TEMPORARY: downloads gradients to CPU for sorting. Will be
   replaced by a GPU thrust::sort implementation in a follow-up. *)
let sparse_grad_mask params top_frac =
  let total = Array.fold_left (fun acc p ->
    acc + Gpu.numel p.Tensor.grad) 0 params in
  let k = max 1 (int_of_float (float_of_int total *. top_frac)) in
  (* Download all gradients to find threshold *)
  let abs_grads = Array.create_float total in
  let offset = ref 0 in
  Array.iter (fun p ->
    let g = Gpu.download p.Tensor.grad in
    let n = Array.length g in
    for i = 0 to n - 1 do
      abs_grads.(!offset + i) <- Float.abs g.(i)
    done;
    offset := !offset + n
  ) params;
  Array.sort (fun a b -> Float.compare b a) abs_grads;
  let threshold = abs_grads.(k - 1) in
  (* Apply threshold: zero gradients below threshold, re-upload *)
  let kept = ref 0 in
  Array.iter (fun p ->
    let g = Gpu.download p.Tensor.grad in
    for i = 0 to Array.length g - 1 do
      if Float.abs g.(i) >= threshold then incr kept
      else g.(i) <- 0.0
    done;
    Gpu.upload g p.Tensor.grad
  ) params;
  !kept

(* Clip global gradient norm. Downloads to CPU for norm computation,
   then scales on GPU if needed.
   TEMPORARY: CPU-side norm. Will be GPU reduction later. *)
let clip_grad_norm params =
  let norm_sq = ref 0.0 in
  Array.iter (fun p ->
    let g = Gpu.download p.Tensor.grad in
    Array.iter (fun v -> norm_sq := !norm_sq +. v *. v) g
  ) params;
  let norm = sqrt !norm_sq in
  if norm > max_grad_norm then begin
    let scale = max_grad_norm /. norm in
    Array.iter (fun p ->
      let g = Gpu.download p.Tensor.grad in
      let scaled = Array.map (fun v -> v *. scale) g in
      Gpu.upload scaled p.Tensor.grad
    ) params
  end

(* ── Adam step ───────────────────────────────────────────────────── *)

(* One Adam step with scheduled LR. Dispatches to GPU kernel
   per parameter tensor. *)
let adam_step params adam step num_steps =
  let lr_t = get_lr step num_steps in
  let bc1 = 1.0 /. (1.0 -. beta1 ** float_of_int (step + 1)) in
  let bc2 = 1.0 /. (1.0 -. beta2 ** float_of_int (step + 1)) in
  Array.iteri (fun i p ->
    let n = Gpu.numel p.Tensor.data in
    Gpu.adam_step p.Tensor.data p.Tensor.grad
      adam.m.(i) adam.v.(i)
      lr_t beta1 beta2 bc1 bc2 n
  ) params

(* Adam step with fixed LR for interactive RL fine-tuning. *)
let adam_step_fixed params adam step lr =
  let bc1 = 1.0 /. (1.0 -. beta1 ** float_of_int (step + 1)) in
  let bc2 = 1.0 /. (1.0 -. beta2 ** float_of_int (step + 1)) in
  Array.iteri (fun i p ->
    let n = Gpu.numel p.Tensor.data in
    Gpu.adam_step p.Tensor.data p.Tensor.grad
      adam.m.(i) adam.v.(i)
      lr beta1 beta2 bc1 bc2 n
  ) params

(* ── Elastic weight consolidation ────────────────────────────────── *)

(* Pull weights toward anchor (base model values).
   The anchor is a per-parameter GPU buffer saved before RL begins.
   alpha controls pull strength: 0.01 = gentle, 0.1 = strong.

   This is the other half of our memory mechanism. Sparse gradient
   masking selects WHICH weights change. Elastic pull controls
   HOW MUCH they're allowed to drift from baseline. Together they
   let the model accumulate persistent memories without forgetting. *)
let elastic_pull params anchor alpha =
  Array.iteri (fun i p ->
    let n = Gpu.numel p.Tensor.data in
    Gpu.elastic_pull p.Tensor.data anchor.(i) alpha n
  ) params

(* Save anchor: copy each parameter's current values to a GPU buffer. *)
let save_anchor params =
  Array.map (fun p ->
    let n = Gpu.numel p.Tensor.data in
    let buf = Gpu.create n in
    Gpu.copy p.Tensor.data buf n;
    buf
  ) params

(* ── Loss computation ────────────────────────────────────────────── *)

(* NLL loss for one document. Returns (loss_node, seq_len). *)
let compute_loss model tokens =
  Tensor.training := true;
  let seq_len = min Model.block_size (Array.length tokens - 1) in
  let input_tokens = Array.sub tokens 0 seq_len in
  let logits = Forward.gpt_forward_batch model input_tokens seq_len in
  let losses = Array.init seq_len (fun i ->
    let logits_i = Tensor.row logits i in
    let probs_i = Tensor.softmax logits_i in
    Tensor.nll probs_i tokens.(i + 1)
  ) in
  (Tensor.mean losses, seq_len)

(* Contrastive loss (simplified DPO). Pushes probability toward
   chosen response and away from rejected. *)
let contrastive_loss model chosen_tokens rejected_tokens beta =
  Tensor.training := true;
  let forward_nll tokens =
    let len = min Model.block_size (Array.length tokens - 1) in
    let input = Array.sub tokens 0 len in
    let logits = Forward.gpt_forward_batch model input len in
    let nll_per_pos = Array.init len (fun i ->
      Tensor.row logits i |> Tensor.softmax |> fun p ->
      Tensor.nll p tokens.(i + 1)
    ) in
    Tensor.mean nll_per_pos
  in
  let chosen_loss = forward_nll chosen_tokens in
  let rejected_loss = forward_nll rejected_tokens in
  (* Download scalar losses to CPU for margin computation *)
  let log_pi_chosen = -. (Gpu.download chosen_loss.Tensor.data).(0) in
  let log_pi_rejected = -. (Gpu.download rejected_loss.Tensor.data).(0) in
  let margin = beta *. (log_pi_chosen -. log_pi_rejected) in
  let sigmoid_m = 1.0 /. (1.0 +. exp (-. margin)) in
  let scale = (1.0 -. sigmoid_m) *. beta in
  (* Seed gradients and backprop *)
  Gpu.upload [| scale |] chosen_loss.Tensor.grad;
  Tensor.backward chosen_loss;
  Gpu.upload [| -. scale |] rejected_loss.Tensor.grad;
  Tensor.backward rejected_loss;
  let loss_val = -. log (sigmoid_m +. 1e-10) in
  (loss_val, margin)

(* ── Pre-tokenization ────────────────────────────────────────────── *)

let pre_tokenize docs tok =
  let t0 = Sys.time () in
  let tokenized = Array.map (fun doc -> Bpe.encode tok doc) docs in
  Printf.printf "pre-tokenized %d docs in %.2fs\n%!"
    (Array.length docs) (Sys.time () -. t0);
  tokenized

(* ── Checkpoint save / load ──────────────────────────────────────── *)

(* Save: download GPU params to CPU float64, marshal to file.
   This preserves compatibility with the existing format while
   data lives on GPU during training. *)
let save_checkpoint filename params =
  let oc = open_out_bin filename in
  let data_arrays = Array.map (fun (p : Tensor.value) ->
    Gpu.download p.data
  ) params in
  Marshal.to_channel oc data_arrays [Marshal.No_sharing];
  close_out oc;
  let total = Array.fold_left (fun acc a -> acc + Array.length a) 0 data_arrays in
  Printf.printf "saved checkpoint to %s (%d params)\n%!" filename total

(* Load: read CPU float64 arrays, upload to GPU params. *)
let load_checkpoint filename params =
  let ic = open_in_bin filename in
  let data_arrays : float array array = Marshal.from_channel ic in
  close_in ic;
  if Array.length data_arrays <> Array.length params then
    failwith (Printf.sprintf "checkpoint has %d param tensors, model has %d"
      (Array.length data_arrays) (Array.length params));
  Array.iteri (fun i saved ->
    let n_saved = Array.length saved in
    let n_model = Gpu.numel params.(i).Tensor.data in
    if n_saved <> n_model then
      failwith (Printf.sprintf "param %d: checkpoint has %d elements, model has %d"
        i n_saved n_model);
    Gpu.upload saved params.(i).Tensor.data
  ) data_arrays;
  let total = Array.fold_left (fun acc a -> acc + Array.length a) 0 data_arrays in
  Printf.printf "loaded checkpoint from %s (%d params)\n%!" filename total

(* ── Resumable book training checkpoints ─────────────────────────── *)

type book_progress = {
  bp_step : int;
  bp_epoch : int;
  bp_n_docs : int;
  bp_n_steps : int;
  bp_book_hash : string;
  bp_order : int array;
}

let book_hash filename n_docs =
  let ic = open_in filename in
  let len = in_channel_length ic in
  let prefix = really_input_string ic (min 100 len) in
  close_in ic;
  Printf.sprintf "%s|%d|%s" filename n_docs prefix

(* Save book checkpoint. Adam state is downloaded from GPU. *)
let save_book_checkpoint dir params adam progress ~save_adam =
  (if not (Sys.file_exists dir) then
    ignore (Sys.command (Printf.sprintf "mkdir -p %s" dir)));
  save_checkpoint (Filename.concat dir "weights.bin") params;
  if save_adam then begin
    let oc = open_out_bin (Filename.concat dir "adam.bin") in
    let m_cpu = Array.map Gpu.download adam.m in
    let v_cpu = Array.map Gpu.download adam.v in
    Marshal.to_channel oc (m_cpu, v_cpu) [Marshal.No_sharing];
    close_out oc;
    Printf.printf "saved adam state to %s/adam.bin\n%!" dir
  end;
  let oc = open_out_bin (Filename.concat dir "progress.bin") in
  Marshal.to_channel oc progress [];
  close_out oc;
  let oc = open_out (Filename.concat dir "progress.txt") in
  Printf.fprintf oc "step: %d / %d\nepoch: %d\nn_docs: %d\nbook_hash: %s\n"
    progress.bp_step progress.bp_n_steps progress.bp_epoch
    progress.bp_n_docs progress.bp_book_hash;
  close_out oc

(* Load book checkpoint. Adam state is uploaded to GPU. *)
let load_book_checkpoint dir params =
  let progress_file = Filename.concat dir "progress.bin" in
  if not (Sys.file_exists progress_file) then None
  else begin
    load_checkpoint (Filename.concat dir "weights.bin") params;
    let ic = open_in_bin progress_file in
    let progress : book_progress = Marshal.from_channel ic in
    close_in ic;
    let adam_file = Filename.concat dir "adam.bin" in
    let adam =
      if Sys.file_exists adam_file then begin
        let ic = open_in_bin adam_file in
        let (m_cpu, v_cpu) : float array array * float array array =
          Marshal.from_channel ic in
        close_in ic;
        Printf.printf "loaded adam state from %s/adam.bin\n%!" dir;
        let m = Array.map (fun a ->
          let buf = Gpu.create (Array.length a) in
          Gpu.upload a buf; buf) m_cpu in
        let v = Array.map (fun a ->
          let buf = Gpu.create (Array.length a) in
          Gpu.upload a buf; buf) v_cpu in
        { m; v }
      end else begin
        Printf.printf "no adam state found, starting with fresh optimizer\n%!";
        init_adam params
      end
    in
    Some (adam, progress)
  end

(* ── Main training loop ──────────────────────────────────────────── *)

let format_duration secs =
  let h = int_of_float secs / 3600 in
  let m = (int_of_float secs mod 3600) / 60 in
  let s = int_of_float secs mod 60 in
  if h > 0 then Printf.sprintf "%dh%02dm%02ds" h m s
  else if m > 0 then Printf.sprintf "%dm%02ds" m s
  else Printf.sprintf "%ds" s

let train model params tokenized_docs num_steps
    ?(start_step=0) ?(checkpoint_base="") () =
  let adam = init_adam params in
  let loss_sum = ref 0.0 in
  let t_start = Sys.time () in
  for step = start_step to num_steps - 1 do
    let tokens = tokenized_docs.(step mod Array.length tokenized_docs) in
    (* Force GC between steps to free autograd graph from previous step.
       Without this, the GC may free GPU buffers mid-computation via
       finalizers, causing use-after-free on device memory. *)
    Gc.full_major ();
    let (loss, _) = compute_loss model tokens in
    let loss_val = (Gpu.download loss.Tensor.data).(0) in
    loss_sum := !loss_sum +. loss_val;
    Tensor.backward loss;
    clip_grad_norm params;
    adam_step params adam step num_steps;
    let log_interval = 500 in
    if (step + 1) mod log_interval = 0 then begin
      let elapsed = Sys.time () -. t_start in
      let steps_done = float_of_int (step - start_step + 1) in
      let steps_left = float_of_int (num_steps - step - 1) in
      let eta = elapsed *. steps_left /. steps_done in
      Printf.printf "step %5d / %d | loss %.4f | %s elapsed | %s remaining\n%!"
        (step + 1) num_steps (!loss_sum /. float_of_int log_interval)
        (format_duration elapsed) (format_duration eta);
      loss_sum := 0.0
    end;
    if checkpoint_base <> "" && (step + 1) mod 2500 = 0
       && (step + 1) < num_steps then begin
      let filename = Printf.sprintf "%s_%dk.bin"
        checkpoint_base ((step + 1) / 1000) in
      save_checkpoint filename params
    end
  done
