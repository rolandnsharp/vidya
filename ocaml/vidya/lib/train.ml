(* train.ml — Adam optimizer, LR schedule, training loop, checkpoints
   ====================================================================

   Adam with cosine LR schedule, warmup, and gradient clipping.
   Lower peak LR (0.001) stretched over 200K steps for the 10M
   param model. Longer warmup (2000 steps) for stability.
   Gradient clipping at norm 1.0 prevents explosions.

   Running average loss over 2500-step windows shows the real trend
   (per-doc loss bounces 2.4-4.6 depending on doc difficulty). *)

type adam_state = { m : float array; v : float array }

let learning_rate = 0.001
let beta1 = 0.9
let beta2 = 0.999
let eps_adam = 1e-8
let max_grad_norm = 1.0
let warmup_steps = 2000

let init_adam params =
  let total = Array.fold_left (fun acc p -> acc + Array.length p.Tensor.data) 0 params in
  { m = Array.make total 0.0; v = Array.make total 0.0 }

(* Cosine LR schedule with linear warmup. *)
let get_lr step num_steps =
  if step < warmup_steps then
    learning_rate *. float_of_int step /. float_of_int warmup_steps
  else
    let progress =
      float_of_int (step - warmup_steps)
      /. float_of_int (num_steps - warmup_steps) in
    learning_rate *. 0.5 *. (1.0 +. cos (Float.pi *. progress))

(* Clip gradient norm to max_grad_norm. *)
let clip_grad_norm params =
  let norm_sq = ref 0.0 in
  Array.iter (fun p ->
    for i = 0 to Array.length p.Tensor.grad - 1 do
      norm_sq := !norm_sq +. p.Tensor.grad.(i) *. p.Tensor.grad.(i)
    done
  ) params;
  let norm = sqrt !norm_sq in
  if norm > max_grad_norm then begin
    let scale = max_grad_norm /. norm in
    Array.iter (fun p ->
      for i = 0 to Array.length p.Tensor.grad - 1 do
        p.Tensor.grad.(i) <- p.Tensor.grad.(i) *. scale
      done
    ) params
  end

(* One Adam step: update parameters, zero gradients. *)
let adam_step params adam step num_steps =
  let lr_t = get_lr step num_steps in
  let bc1 = 1.0 /. (1.0 -. beta1 ** float_of_int (step + 1)) in
  let bc2 = 1.0 /. (1.0 -. beta2 ** float_of_int (step + 1)) in
  let offset = ref 0 in
  params |> Array.iter (fun p ->
    let n = Array.length p.Tensor.data in
    for i = 0 to n - 1 do
      let fi = !offset + i in
      adam.m.(fi) <- beta1 *. adam.m.(fi) +. (1.0 -. beta1) *. p.Tensor.grad.(i);
      adam.v.(fi) <- beta2 *. adam.v.(fi) +. (1.0 -. beta2) *. p.Tensor.grad.(i) *. p.Tensor.grad.(i);
      let m_hat = adam.m.(fi) *. bc1 in
      let v_hat = adam.v.(fi) *. bc2 in
      p.Tensor.data.(i) <- p.Tensor.data.(i) -. lr_t *. m_hat /. (sqrt v_hat +. eps_adam);
      p.Tensor.grad.(i) <- 0.0
    done;
    offset := !offset + n)

(* Compute NLL loss for one document. *)
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

(* Pre-tokenize all docs at startup. Saves ~500 hash lookups × doc_len
   per training step. *)
let pre_tokenize docs tok =
  let t0 = Sys.time () in
  let tokenized = Array.map (fun doc ->
    Bpe.encode tok doc
  ) docs in
  Printf.printf "pre-tokenized %d docs in %.2fs\n%!"
    (Array.length docs) (Sys.time () -. t0);
  tokenized

(* Main training loop. *)
let train model params tokenized_docs num_steps =
  let adam = init_adam params in
  let loss_sum = ref 0.0 in
  for step = 0 to num_steps - 1 do
    let tokens = tokenized_docs.(step mod Array.length tokenized_docs) in
    let (loss, _) = compute_loss model tokens in
    loss_sum := !loss_sum +. loss.Tensor.data.(0);
    Tensor.backward loss;
    clip_grad_norm params;
    adam_step params adam step num_steps;
    if (step + 1) mod 2500 = 0 then begin
      Printf.printf "step %5d / %5d | loss %.4f\n%!"
        (step + 1) num_steps (!loss_sum /. 2500.0);
      loss_sum := 0.0
    end
  done

(* ── Checkpoint save / load ───────────────────────────────────────── *)

(* Save all parameter data arrays to a binary file using Marshal.
   Format: Marshal'd float array array (one entry per param tensor). *)
let save_checkpoint filename params =
  let oc = open_out_bin filename in
  let data_arrays = Array.map (fun (p : Tensor.value) -> p.data) params in
  Marshal.to_channel oc data_arrays [Marshal.No_sharing];
  close_out oc;
  let total = Array.fold_left (fun acc (p : Tensor.value) -> acc + Array.length p.data) 0 params in
  Printf.printf "saved checkpoint to %s (%d params)\n%!" filename total

(* Load saved weights into live parameter tensors. Validates counts
   and sizes match before blitting. *)
let load_checkpoint filename params =
  let ic = open_in_bin filename in
  let data_arrays : float array array = Marshal.from_channel ic in
  close_in ic;
  if Array.length data_arrays <> Array.length params then
    failwith (Printf.sprintf "checkpoint has %d param tensors, model has %d"
      (Array.length data_arrays) (Array.length params));
  Array.iteri (fun i saved ->
    if Array.length saved <> Array.length params.(i).Tensor.data then
      failwith (Printf.sprintf "param %d: checkpoint has %d elements, model has %d"
        i (Array.length saved) (Array.length params.(i).Tensor.data));
    Array.blit saved 0 params.(i).Tensor.data 0 (Array.length saved)
  ) data_arrays;
  let total = Array.fold_left (fun acc a -> acc + Array.length a) 0 data_arrays in
  Printf.printf "loaded checkpoint from %s (%d params)\n%!" filename total
