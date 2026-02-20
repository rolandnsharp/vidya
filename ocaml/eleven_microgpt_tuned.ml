(* eleven_microgpt_tuned.ml — Stage 11: Training Improvements
   ================================================================

   Based on Stage 10 (scaled up, 1.3M params, 298s).

   WHAT CHANGES
   ============
   Stage 10 produced recognizable Plotinus but the loss curve stalled:
   it hit 2.42 at step 5000 then bounced back to 2.81 at step 10000.
   The cosine LR schedule decayed too fast — by step 8000 the LR was
   ~0.0001, barely learning. Four improvements:

   1. WEIGHT TYING (quality)
      Share the embedding matrix (wte) with the output projection
      (lm_head). Both are [vocab_size, n_embd]. This is standard in
      GPT-2 and most modern LLMs. Benefits:
      - The output layer directly benefits from embedding learning
      - Saves 74K params (1.3M → 1.25M) — less to optimize
      - Better gradient signal to the embedding matrix

   2. RESIDUAL SCALING (stability)
      Scale the output of each residual branch (attn_wo, mlp_fc2) by
      1/sqrt(2 * n_layer) at initialization. With 6 layers, that's
      12 residual additions. Without scaling, variance grows ~12x
      through the network. With scaling, it stays bounded.
      GPT-2 uses this exact technique.

   3. HIGHER LR + MORE STEPS (convergence)
      LR: 0.001 → 0.003, steps: 10K → 100K, warmup: 200 → 400.
      The cosine schedule now stretches over 20K steps, so the model
      keeps learning well past the halfway point. Peak LR 0.003 is
      aggressive but the gradient clipping (norm 1.0) keeps it safe.

   4. RUNNING AVERAGE LOSS (monitoring)
      Instead of per-doc loss (which bounces 2.4-4.6), report the
      average loss over the last 500 steps. Shows the real trend.

   Architecture unchanged: 128-dim, 8 heads, 6 layers, 128 context.

   Compile:
     ocamlopt -O2 -o microgpt_tuned \
       blas_stubs.c eleven_microgpt_tuned.ml \
       -ccopt "-I/usr/include/x86_64-linux-gnu" -cclib -lopenblas

   Run:
     ./microgpt_tuned *)

let () = Random.init 42

(* ══════════════════════════════════════════════════════════════════════
   BLAS FFI — 6 dgemm variants via 3-bit op flag
   ══════════════════════════════════════════════════════════════════════ *)
external blas_dgemm
  : int -> int -> int -> int
    -> float array -> float array -> float array
    -> unit
  = "caml_dgemm_byte" "caml_dgemm"

(* ══════════════════════════════════════════════════════════════════════
   UTILITIES
   ══════════════════════════════════════════════════════════════════════ *)

let random_gauss ?(mean = 0.0) ?(std = 1.0) () =
  let rec sample () =
    let u = Random.float 2.0 -. 1.0 in
    let v = Random.float 2.0 -. 1.0 in
    let s = u *. u +. v *. v in
    if s >= 1.0 || s = 0.0 then sample ()
    else mean +. std *. u *. sqrt (-2.0 *. log s /. s)
  in
  sample ()

let shuffle arr =
  for i = Array.length arr - 1 downto 1 do
    let j = Random.int (i + 1) in
    let tmp = arr.(i) in
    arr.(i) <- arr.(j);
    arr.(j) <- tmp
  done

let weighted_choice weights =
  let total = Array.fold_left (+.) 0.0 weights in
  let r = Random.float total in
  Array.fold_left (fun (chosen, remaining) w ->
    if remaining <= 0.0 then (chosen, remaining)
    else if remaining -. w <= 0.0 then (chosen, remaining -. w)
    else (chosen + 1, remaining -. w)
  ) (0, r) weights
  |> fst

let load_docs filename =
  if not (Sys.file_exists filename) then begin
    Printf.eprintf "Error: %s not found.\n" filename;
    exit 1
  end;
  let ic = open_in filename in
  let docs = ref [] in
  (try while true do
    let line = input_line ic |> String.trim in
    if String.length line > 0 then docs := line :: !docs
  done with End_of_file -> ());
  close_in ic;
  let arr = !docs |> List.rev |> Array.of_list in
  shuffle arr;
  arr

(* ══════════════════════════════════════════════════════════════════════
   BPE TOKENIZER — from scratch

   Byte-Pair Encoding learns a vocabulary of subword tokens by
   iteratively merging the most frequent adjacent pair in the corpus.

   Data structures:
     vocab     : string array   — token_id → token string (for decode)
     merges    : (int*int, int) Hashtbl — (left_id, right_id) → merged_id
     char_to_id: (char, int) Hashtbl   — character → initial token ID

   The merge priority is implicit: lower merged_id = learned earlier =
   higher priority. During encoding, we always apply the highest-priority
   (lowest ID) merge first. This matches GPT-2's BPE convention.
   ══════════════════════════════════════════════════════════════════════ *)

let n_merges = 500

(* bpe_train: Learn BPE merges from the training corpus.
   1. Build char vocab from all unique characters in docs
   2. Tokenize entire corpus as char IDs (flat array, -1 as doc separator)
   3. For each merge round: count pairs, merge most frequent
   Returns (vocab, merges, char_to_id, bos_id, vocab_size) *)
let bpe_train docs n_merges =
  (* Step 1: Collect unique characters and assign initial IDs *)
  let char_set = Hashtbl.create 128 in
  Array.iter (fun doc ->
    String.iter (fun ch -> Hashtbl.replace char_set ch ()) doc
  ) docs;
  let chars =
    Hashtbl.fold (fun ch () acc -> ch :: acc) char_set []
    |> List.sort Char.compare
    |> Array.of_list in
  let n_chars = Array.length chars in
  let char_to_id = Hashtbl.create n_chars in
  Array.iteri (fun i ch -> Hashtbl.add char_to_id ch i) chars;

  (* vocab array: maps token ID → surface string.
     IDs 0..n_chars-1 are single characters.
     IDs n_chars..n_chars+n_merges-1 are merged tokens.
     ID n_chars+n_merges is BOS. *)
  let vocab = Array.make (n_chars + n_merges + 1) "" in
  Array.iteri (fun i ch -> vocab.(i) <- String.make 1 ch) chars;

  (* Step 2: Build flat corpus array with -1 separators between docs.
     The separator prevents merges across document boundaries. *)
  let total_len = ref 0 in
  Array.iter (fun doc ->
    if !total_len > 0 then incr total_len;  (* separator *)
    total_len := !total_len + String.length doc
  ) docs;
  let corpus = Array.make !total_len (-1) in
  let pos = ref 0 in
  Array.iter (fun doc ->
    if !pos > 0 then begin corpus.(!pos) <- -1; incr pos end;
    String.iter (fun ch ->
      corpus.(!pos) <- Hashtbl.find char_to_id ch;
      incr pos
    ) doc
  ) docs;
  let corpus_len = ref !total_len in

  (* Step 3: Iteratively merge the most frequent pair.

     OPTIMIZATION: Use a flat int array for pair counts instead of
     Hashtbl. The Hashtbl approach was 33s because every pair lookup
     boxes an (int, int) tuple on the heap, hashes it, and does
     polymorphic comparison. A flat array indexed by a*max_id+b
     avoids all that overhead.

     Array size: max_id² = 580² = 336,400 ints ≈ 2.7MB.
     Zeroing it each round via Array.fill is a single fast memset. *)
  let max_id = n_chars + n_merges + 1 in
  let pair_counts = Array.make (max_id * max_id) 0 in
  let merges = Hashtbl.create (n_merges * 2) in
  let actual_merges = ref 0 in
  (try
    for merge_round = 0 to n_merges - 1 do
      (* Zero pair counts *)
      Array.fill pair_counts 0 (max_id * max_id) 0;

      (* Count all adjacent pairs (skip separator tokens = -1) *)
      for i = 0 to !corpus_len - 2 do
        let a = corpus.(i) and b = corpus.(i + 1) in
        if a >= 0 && b >= 0 then begin
          let idx = a * max_id + b in
          pair_counts.(idx) <- pair_counts.(idx) + 1
        end
      done;

      (* Find the most frequent pair *)
      let best_a = ref 0 and best_b = ref 0 in
      let best_count = ref 0 in
      for i = 0 to !corpus_len - 2 do
        let a = corpus.(i) and b = corpus.(i + 1) in
        if a >= 0 && b >= 0 then begin
          let c = pair_counts.(a * max_id + b) in
          if c > !best_count then begin
            best_count := c; best_a := a; best_b := b
          end
        end
      done;

      if !best_count < 2 then
        raise Exit  (* no useful merges left *)
      else begin
        let a = !best_a and b = !best_b in
        let new_id = n_chars + merge_round in
        Hashtbl.replace merges (a, b) new_id;
        vocab.(new_id) <- vocab.(a) ^ vocab.(b);
        actual_merges := merge_round + 1;

        (* Replace all occurrences of (a, b) with new_id in corpus.
           In-place: j is the write cursor, always <= i (read cursor).
           This is safe because merges only shrink the sequence. *)
        let j = ref 0 in
        let i = ref 0 in
        while !i < !corpus_len do
          if !i < !corpus_len - 1
             && corpus.(!i) = a && corpus.(!i + 1) = b then begin
            corpus.(!j) <- new_id;
            incr j;
            i := !i + 2
          end else begin
            corpus.(!j) <- corpus.(!i);
            incr j;
            incr i
          end
        done;
        corpus_len := !j
      end
    done
  with Exit -> ());

  let bos_id = n_chars + n_merges in
  vocab.(bos_id) <- "<BOS>";
  let vocab_size = bos_id + 1 in

  (* Print BPE training stats.
     Compression ratio: compare original char count to post-merge token count.
     The corpus array has n_docs-1 separators (-1); subtract those. *)
  let n_docs = Array.length docs in
  let n_separators = max 0 (n_docs - 1) in
  let orig_chars = !total_len - n_separators in
  let final_tokens = !corpus_len - n_separators in
  let ratio = float_of_int orig_chars /. float_of_int (max 1 final_tokens) in
  Printf.printf "BPE: %d chars + %d merges = %d vocab | %.1f chars/token\n%!"
    n_chars !actual_merges vocab_size ratio;

  (vocab, merges, char_to_id, bos_id, vocab_size)

(* bpe_encode: Tokenize a string using learned BPE merges.
   1. Convert chars to initial token IDs
   2. Repeatedly apply highest-priority merge (lowest merged_id)
   3. Wrap with BOS (prepend and append)

   For a typical doc (~100-200 chars), this runs in under 1μs.
   The key insight: we always pick the merge with the smallest new_id
   (= learned first = most frequent in training data). This ensures
   deterministic tokenization matching the training distribution. *)
let bpe_encode merges char_to_id bos_id doc =
  let n = String.length doc in
  if n = 0 then [|bos_id; bos_id|]
  else begin
    (* Convert string to char-level token IDs *)
    let tokens = Array.init n (fun i -> Hashtbl.find char_to_id doc.[i]) in
    let len = ref n in

    (* Repeatedly find and apply the highest-priority merge *)
    let changed = ref true in
    while !changed do
      changed := false;
      (* Scan for the pair whose merged_id is smallest (= highest priority) *)
      let best_mid = ref max_int in
      let best_a = ref 0 and best_b = ref 0 in
      for i = 0 to !len - 2 do
        match Hashtbl.find_opt merges (tokens.(i), tokens.(i + 1)) with
        | Some mid when mid < !best_mid ->
          best_mid := mid;
          best_a := tokens.(i);
          best_b := tokens.(i + 1)
        | _ -> ()
      done;

      if !best_mid < max_int then begin
        (* Apply this merge everywhere in the token sequence *)
        let a = !best_a and b = !best_b and new_id = !best_mid in
        let j = ref 0 in
        let i = ref 0 in
        while !i < !len do
          if !i < !len - 1
             && tokens.(!i) = a && tokens.(!i + 1) = b then begin
            tokens.(!j) <- new_id;
            incr j;
            i := !i + 2
          end else begin
            tokens.(!j) <- tokens.(!i);
            incr j;
            incr i
          end
        done;
        len := !j;
        changed := true
      end
    done;

    (* Wrap with BOS: [BOS, token1, token2, ..., tokenN, BOS] *)
    let result = Array.make (!len + 2) bos_id in
    Array.blit tokens 0 result 1 !len;
    result
  end

(* ══════════════════════════════════════════════════════════════════════
   TENSOR AUTOGRAD ENGINE (unchanged)
   ══════════════════════════════════════════════════════════════════════ *)

let next_id = ref 0
let fresh_id () = let id = !next_id in incr next_id; id

type value = {
  id : int;
  data : float array;
  grad : float array;
  shape : int array;
  children : value array;
  backward_fn : unit -> unit;
}

let make_param shape data =
  { id = fresh_id (); data; grad = Array.make (Array.length data) 0.0;
    shape; children = [||]; backward_fn = (fun () -> ()) }

let dummy_node = { id = -1; data = [||]; grad = [||]; shape = [||];
                   children = [||]; backward_fn = (fun () -> ()) }

(* ══════════════════════════════════════════════════════════════════════
   SHARED OPS — element-wise, work for any shape
   ══════════════════════════════════════════════════════════════════════ *)

let tensor_add a b =
  let n = Array.length a.data in
  let y_data = Array.create_float n in
  for i = 0 to n - 1 do y_data.(i) <- a.data.(i) +. b.data.(i) done;
  let y_grad = Array.make n 0.0 in
  let backward () =
    for i = 0 to n - 1 do
      a.grad.(i) <- a.grad.(i) +. y_grad.(i);
      b.grad.(i) <- b.grad.(i) +. y_grad.(i)
    done in
  { id = fresh_id (); data = y_data; grad = y_grad;
    shape = [|n|]; children = [|a; b|]; backward_fn = backward }

let tensor_scale x s =
  let n = Array.length x.data in
  let y_data = Array.create_float n in
  for i = 0 to n - 1 do y_data.(i) <- x.data.(i) *. s done;
  let y_grad = Array.make n 0.0 in
  let backward () =
    for i = 0 to n - 1 do
      x.grad.(i) <- x.grad.(i) +. y_grad.(i) *. s
    done in
  { id = fresh_id (); data = y_data; grad = y_grad;
    shape = [|n|]; children = [|x|]; backward_fn = backward }

let tensor_gelu x =
  let n = Array.length x.data in
  let a = 0.7978845608028654 in
  let b = 0.044715 in
  let y_data = Array.create_float n in
  for i = 0 to n - 1 do
    let xi = x.data.(i) in
    let inner = a *. (xi +. b *. xi *. xi *. xi) in
    let t = tanh inner in
    y_data.(i) <- 0.5 *. xi *. (1.0 +. t)
  done;
  let y_grad = Array.make n 0.0 in
  let backward () =
    for i = 0 to n - 1 do
      let xi = x.data.(i) in
      let inner = a *. (xi +. b *. xi *. xi *. xi) in
      let t = tanh inner in
      let dtanh = 1.0 -. t *. t in
      let dgelu = 0.5 *. (1.0 +. t)
        +. 0.5 *. xi *. dtanh *. a *. (1.0 +. 3.0 *. b *. xi *. xi) in
      x.grad.(i) <- x.grad.(i) +. y_grad.(i) *. dgelu
    done in
  { id = fresh_id (); data = y_data; grad = y_grad;
    shape = [|n|]; children = [|x|]; backward_fn = backward }

let tensor_softmax x =
  let n = Array.length x.data in
  let max_val = ref neg_infinity in
  for i = 0 to n - 1 do
    if x.data.(i) > !max_val then max_val := x.data.(i)
  done;
  let y_data = Array.create_float n in
  let sum_exp = ref 0.0 in
  for i = 0 to n - 1 do
    let e = exp (x.data.(i) -. !max_val) in
    y_data.(i) <- e;
    sum_exp := !sum_exp +. e
  done;
  let inv_sum = 1.0 /. !sum_exp in
  for i = 0 to n - 1 do y_data.(i) <- y_data.(i) *. inv_sum done;
  let y_grad = Array.make n 0.0 in
  let backward () =
    let dot = ref 0.0 in
    for j = 0 to n - 1 do dot := !dot +. y_grad.(j) *. y_data.(j) done;
    for i = 0 to n - 1 do
      x.grad.(i) <- x.grad.(i) +. y_data.(i) *. (y_grad.(i) -. !dot)
    done in
  { id = fresh_id (); data = y_data; grad = y_grad;
    shape = [|n|]; children = [|x|]; backward_fn = backward }

let tensor_nll probs target =
  let y_data = [| -. log probs.data.(target) |] in
  let y_grad = [| 0.0 |] in
  let backward () =
    probs.grad.(target) <- probs.grad.(target)
      -. y_grad.(0) /. probs.data.(target) in
  { id = fresh_id (); data = y_data; grad = y_grad;
    shape = [|1|]; children = [|probs|]; backward_fn = backward }

let tensor_mean losses =
  let n = Array.length losses in
  let nf = float_of_int n in
  let total = Array.fold_left (fun acc l -> acc +. l.data.(0)) 0.0 losses in
  let y_data = [| total /. nf |] in
  let y_grad = [| 0.0 |] in
  let backward () =
    Array.iter (fun l ->
      l.grad.(0) <- l.grad.(0) +. y_grad.(0) /. nf
    ) losses in
  { id = fresh_id (); data = y_data; grad = y_grad;
    shape = [|1|]; children = Array.copy losses; backward_fn = backward }

let tensor_row mat row_idx =
  let cols = mat.shape.(1) in
  let off = row_idx * cols in
  let y_data = Array.create_float cols in
  for j = 0 to cols - 1 do y_data.(j) <- mat.data.(off + j) done;
  let y_grad = Array.make cols 0.0 in
  let backward () =
    for j = 0 to cols - 1 do
      mat.grad.(off + j) <- mat.grad.(off + j) +. y_grad.(j)
    done in
  { id = fresh_id (); data = y_data; grad = y_grad;
    shape = [|cols|]; children = [|mat|]; backward_fn = backward }

(* ══════════════════════════════════════════════════════════════════════
   SINGLE-VECTOR OPS — inference only (one token at a time with KV cache)
   ══════════════════════════════════════════════════════════════════════ *)

let matmul w x =
  let m = w.shape.(0) and n = w.shape.(1) in
  let y_data = Array.create_float m in
  blas_dgemm 0 m 1 n w.data x.data y_data;
  let y_grad = Array.make m 0.0 in
  let backward () =
    blas_dgemm 1 m n 1 y_grad x.data w.grad;
    blas_dgemm 5 n 1 m w.data y_grad x.grad
  in
  { id = fresh_id (); data = y_data; grad = y_grad;
    shape = [|m|]; children = [|w; x|]; backward_fn = backward }

let tensor_rmsnorm x =
  let n = Array.length x.data in
  let nf = float_of_int n in
  let ms = ref 0.0 in
  for i = 0 to n - 1 do ms := !ms +. x.data.(i) *. x.data.(i) done;
  let rms = sqrt (!ms /. nf +. 1e-5) in
  let y_data = Array.create_float n in
  for i = 0 to n - 1 do y_data.(i) <- x.data.(i) /. rms done;
  let y_grad = Array.make n 0.0 in
  let backward () =
    let dot_gy = ref 0.0 in
    for i = 0 to n - 1 do dot_gy := !dot_gy +. y_grad.(i) *. y_data.(i) done;
    let mean_gy = !dot_gy /. nf in
    for i = 0 to n - 1 do
      x.grad.(i) <- x.grad.(i) +. (y_grad.(i) -. y_data.(i) *. mean_gy) /. rms
    done in
  { id = fresh_id (); data = y_data; grad = y_grad;
    shape = [|n|]; children = [|x|]; backward_fn = backward }

(* ══════════════════════════════════════════════════════════════════════
   BATCHED OPS — training (all positions at once)
   ══════════════════════════════════════════════════════════════════════ *)

let batch_matmul w x =
  let m = w.shape.(0) and n = w.shape.(1) in
  let s = Array.length x.data / n in
  let y_data = Array.create_float (s * m) in
  blas_dgemm 2 s m n x.data w.data y_data;
  let y_grad = Array.make (s * m) 0.0 in
  let backward () =
    blas_dgemm 5 m n s y_grad x.data w.grad;
    blas_dgemm 1 s n m y_grad w.data x.grad
  in
  { id = fresh_id (); data = y_data; grad = y_grad;
    shape = [|s; m|]; children = [|w; x|]; backward_fn = backward }

let batch_rmsnorm ~dim x =
  let s = Array.length x.data / dim in
  let dimf = float_of_int dim in
  let y_data = Array.create_float (s * dim) in
  let rms_vals = Array.create_float s in
  for i = 0 to s - 1 do
    let off = i * dim in
    let ms = ref 0.0 in
    for j = 0 to dim - 1 do
      let v = x.data.(off + j) in
      ms := !ms +. v *. v
    done;
    let rms = sqrt (!ms /. dimf +. 1e-5) in
    rms_vals.(i) <- rms;
    for j = 0 to dim - 1 do
      y_data.(off + j) <- x.data.(off + j) /. rms
    done
  done;
  let y_grad = Array.make (s * dim) 0.0 in
  let backward () =
    for i = 0 to s - 1 do
      let off = i * dim in
      let rms = rms_vals.(i) in
      let dot_gy = ref 0.0 in
      for j = 0 to dim - 1 do
        dot_gy := !dot_gy +. y_grad.(off + j) *. y_data.(off + j)
      done;
      let mean_gy = !dot_gy /. dimf in
      for j = 0 to dim - 1 do
        x.grad.(off + j) <- x.grad.(off + j)
          +. (y_grad.(off + j) -. y_data.(off + j) *. mean_gy) /. rms
      done
    done
  in
  { id = fresh_id (); data = y_data; grad = y_grad;
    shape = [|s; dim|]; children = [|x|]; backward_fn = backward }

let batch_embed wte tokens n_embd =
  let s = Array.length tokens in
  let data = Array.create_float (s * n_embd) in
  for i = 0 to s - 1 do
    Array.blit wte.data (tokens.(i) * n_embd) data (i * n_embd) n_embd
  done;
  let grad = Array.make (s * n_embd) 0.0 in
  let backward () =
    for i = 0 to s - 1 do
      let src = tokens.(i) * n_embd in
      let dst = i * n_embd in
      for j = 0 to n_embd - 1 do
        wte.grad.(src + j) <- wte.grad.(src + j) +. grad.(dst + j)
      done
    done
  in
  { id = fresh_id (); data; grad;
    shape = [|s; n_embd|]; children = [|wte|]; backward_fn = backward }

(* ══════════════════════════════════════════════════════════════════════
   BACKWARD PASS (unchanged)
   ══════════════════════════════════════════════════════════════════════ *)

let topological_sort root =
  let visited = Hashtbl.create 1024 in
  let topo = ref [] in
  let rec build_topo v =
    if not (Hashtbl.mem visited v.id) then begin
      Hashtbl.add visited v.id ();
      Array.iter build_topo v.children;
      topo := v :: !topo
    end
  in
  build_topo root;
  !topo

let backward loss =
  let topo = topological_sort loss in
  loss.grad.(0) <- 1.0;
  List.iter (fun v -> v.backward_fn ()) topo

(* ══════════════════════════════════════════════════════════════════════
   MODEL DEFINITION — weight-tied, residual-scaled
   128-dim, 8 heads, 6 layers, 128-token context, ~1.25M params
   ══════════════════════════════════════════════════════════════════════ *)

let n_layer = 6
let n_embd = 128
let block_size = 128
let n_head = 8
let head_dim = n_embd / n_head   (* = 16 *)
let half_dim = head_dim / 2      (* = 8 *)

let rope_freqs = Array.init half_dim (fun i ->
  1.0 /. (10000.0 ** (float_of_int (2 * i) /. float_of_int head_dim)))

let rope_cos = Array.init block_size (fun pos ->
  Array.init half_dim (fun i ->
    cos (float_of_int pos *. rope_freqs.(i))))

let rope_sin = Array.init block_size (fun pos ->
  Array.init half_dim (fun i ->
    sin (float_of_int pos *. rope_freqs.(i))))

type layer_weights = {
  attn_wq : value; attn_wk : value;
  attn_wv : value; attn_wo : value;
  mlp_fc1 : value; mlp_fc2 : value;
}

type kv_cache = {
  k_cache : float array array;
  v_cache : float array array;
  k_nodes : value array;
  v_nodes : value array;
  mutable len : int;
}

let make_kv_cache () = {
  k_cache = Array.init n_head (fun _ -> Array.create_float (block_size * head_dim));
  v_cache = Array.init n_head (fun _ -> Array.create_float (block_size * head_dim));
  k_nodes = Array.make block_size dummy_node;
  v_nodes = Array.make block_size dummy_node;
  len = 0;
}

type model = {
  wte : value;
  lm_head : value;
  layers : layer_weights array;
}

let init_matrix ?(std = 0.08) nout nin =
  let data = Array.init (nout * nin) (fun _ -> random_gauss ~std ()) in
  make_param [|nout; nin|] data

let init_model vocab_size =
  let wte = init_matrix vocab_size n_embd in
  (* WEIGHT TYING: lm_head shares the same matrix as wte.
     The output projection logit for token t is just the dot product
     of the hidden state with t's embedding vector. This creates a
     tight feedback loop: good embeddings → good predictions → better
     gradients → better embeddings. GPT-2, LLaMA, and most modern
     LLMs use this technique. Saves vocab_size × n_embd params. *)
  let lm_head = wte in
  (* RESIDUAL SCALING: residual branch outputs (attn_wo, mlp_fc2) are
     initialized with std = 0.08 / sqrt(2 * n_layer). Each layer adds
     two residual connections (attention + MLP). Without scaling, the
     variance of the residual stream grows by ~2*n_layer = 12x through
     the network. Scaling the init keeps it bounded at ~1x. *)
  let residual_std = 0.08 /. sqrt (float_of_int (2 * n_layer)) in
  let make_layer () =
    let attn_wq = init_matrix n_embd n_embd in
    let attn_wk = init_matrix n_embd n_embd in
    let attn_wv = init_matrix n_embd n_embd in
    let attn_wo = init_matrix ~std:residual_std n_embd n_embd in
    let mlp_fc1 = init_matrix (4 * n_embd) n_embd in
    let mlp_fc2 = init_matrix ~std:residual_std n_embd (4 * n_embd) in
    { attn_wq; attn_wk; attn_wv; attn_wo; mlp_fc1; mlp_fc2 }
  in
  let layers = Array.init n_layer (fun _ -> make_layer ()) in
  { wte; lm_head; layers }

let collect_params model =
  let layer_params l =
    [l.attn_wq; l.attn_wk; l.attn_wv; l.attn_wo; l.mlp_fc1; l.mlp_fc2] in
  (* Only include wte once — lm_head is weight-tied to wte.
     Including it twice would make Adam apply double updates. *)
  [model.wte]
  @ (model.layers |> Array.to_list |> List.map layer_params |> List.flatten)
  |> Array.of_list

(* ══════════════════════════════════════════════════════════════════════
   CHECKPOINT SAVE / LOAD

   Saves all parameter float arrays to a binary file using OCaml's
   Marshal. Each param's data array is serialized in order:
     wte, layer0.wq, layer0.wk, ..., layer0.fc2, layer1.wq, ...

   With weight tying, lm_head = wte so it's included via wte.
   File size ≈ params × 8 bytes + small header ≈ ~10MB for 1.25M params.

   Usage:
     ./microgpt_tuned            → train from scratch, save checkpoint
     ./microgpt_tuned model.bin  → load checkpoint, skip training
   ══════════════════════════════════════════════════════════════════════ *)

let save_checkpoint filename params =
  let oc = open_out_bin filename in
  let data_arrays = Array.map (fun p -> p.data) params in
  Marshal.to_channel oc data_arrays [Marshal.No_sharing];
  close_out oc;
  Printf.printf "saved checkpoint to %s (%d params)\n%!" filename
    (Array.fold_left (fun acc p -> acc + Array.length p.data) 0 params)

let load_checkpoint filename params =
  let ic = open_in_bin filename in
  let data_arrays : float array array = Marshal.from_channel ic in
  close_in ic;
  if Array.length data_arrays <> Array.length params then
    failwith (Printf.sprintf "checkpoint has %d params, model expects %d"
      (Array.length data_arrays) (Array.length params));
  Array.iteri (fun i p ->
    if Array.length data_arrays.(i) <> Array.length p.data then
      failwith (Printf.sprintf "param %d: checkpoint has %d values, model expects %d"
        i (Array.length data_arrays.(i)) (Array.length p.data));
    Array.blit data_arrays.(i) 0 p.data 0 (Array.length p.data)
  ) params;
  Printf.printf "loaded checkpoint from %s (%d params)\n%!" filename
    (Array.fold_left (fun acc a -> acc + Array.length a) 0 data_arrays)

(* ══════════════════════════════════════════════════════════════════════
   HEAD EXTRACTION / INSERTION HELPERS

   Our tensors are [S, n_embd] with heads interleaved in the inner dim:
     row i, head h = data[i * n_embd + h * head_dim .. + head_dim - 1]

   BLAS needs contiguous [S, head_dim] arrays per head. These helpers
   extract/insert the head slices. Cost: ~1KB of memcpy per call at
   S=64, d=16 — negligible vs the BLAS compute.
   ══════════════════════════════════════════════════════════════════════ *)

(* Extract head h from [S, n_embd] → contiguous [S, head_dim] *)
let extract_head src dst h s =
  let h_off = h * head_dim in
  for i = 0 to s - 1 do
    Array.blit src (i * n_embd + h_off) dst (i * head_dim) head_dim
  done

(* Insert [S, head_dim] into head h of [S, n_embd] — overwrite *)
let insert_head src dst h s =
  let h_off = h * head_dim in
  for i = 0 to s - 1 do
    Array.blit src (i * head_dim) dst (i * n_embd + h_off) head_dim
  done

(* Insert [S, head_dim] into head h of [S, n_embd] — accumulate *)
let insert_head_accum src dst h s =
  let h_off = h * head_dim in
  for i = 0 to s - 1 do
    let d_off = i * n_embd + h_off in
    let s_off = i * head_dim in
    for j = 0 to head_dim - 1 do
      dst.(d_off + j) <- dst.(d_off + j) +. src.(s_off + j)
    done
  done

(* ══════════════════════════════════════════════════════════════════════
   BATCHED TRAINING FORWARD
   ══════════════════════════════════════════════════════════════════════ *)

(* Batched multi-head attention with RoPE, causal mask, and BLAS.

   BLAS operations per head (forward):
     Scores[S,S] = Q_h[S,d] @ K_h[S,d]^T     op=2  (NT overwrite)
     Attn_h[S,d] = Weights[S,S] @ V_h[S,d]    op=0  (NN overwrite)

   BLAS operations per head (backward):
     dWeights[S,S] = dAttn_h[S,d] @ V_h[S,d]^T    op=2
     dV_h[S,d]     = Weights^T[S,S] @ dAttn_h[S,d] op=4  (TN overwrite)
     dQ_rot_h[S,d] = dScores[S,S] @ K_rot_h[S,d]   op=0
     dK_rot_h[S,d] = dScores^T[S,S] @ Q_rot_h[S,d] op=4

   Scalar loops handle: causal mask, softmax forward/backward, RoPE rotation.

   At S=64, d=16: each BLAS call processes 64×64×16 = 65,536 multiply-adds.
   With 4 heads × 6 BLAS calls × 2 (fwd+bwd) = 48 BLAS calls per layer,
   4 layers = 192 additional BLAS calls per step. *)
let batch_fused_attention q k v seq_len =
  (* Apply RoPE rotation to all positions of Q and K *)
  let q_rot = Array.create_float (seq_len * n_embd) in
  let k_rot = Array.create_float (seq_len * n_embd) in
  for pos = 0 to seq_len - 1 do
    let cos_p = rope_cos.(pos) and sin_p = rope_sin.(pos) in
    let row = pos * n_embd in
    for h = 0 to n_head - 1 do
      let base = row + h * head_dim in
      for i = 0 to half_dim - 1 do
        let j0 = base + 2 * i and j1 = base + 2 * i + 1 in
        let c = cos_p.(i) and s = sin_p.(i) in
        q_rot.(j0) <- q.data.(j0) *. c -. q.data.(j1) *. s;
        q_rot.(j1) <- q.data.(j0) *. s +. q.data.(j1) *. c;
        k_rot.(j0) <- k.data.(j0) *. c -. k.data.(j1) *. s;
        k_rot.(j1) <- k.data.(j0) *. s +. k.data.(j1) *. c
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

    (* Extract head data into contiguous arrays *)
    extract_head q_rot q_h h seq_len;
    extract_head k_rot k_h h seq_len;

    (* Scores[S,S] = Q_h[S,d] @ K_h[S,d]^T — BLAS! *)
    blas_dgemm 2 seq_len seq_len head_dim q_h k_h scores;

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
      (* Zero upper triangle — needed for correct BLAS in value weighting *)
      for j = i + 1 to seq_len - 1 do wt.(i * seq_len + j) <- 0.0 done
    done;

    (* Attn_h[S,d] = Weights[S,S] @ V_h[S,d] — BLAS! *)
    extract_head v.data v_h h seq_len;
    blas_dgemm 0 seq_len head_dim seq_len wt v_h attn_h;

    (* Write back to interleaved output *)
    insert_head attn_h out_data h seq_len
  done;

  let out_grad = Array.make (seq_len * n_embd) 0.0 in
  let backward () =
    let dq_rot = Array.make (seq_len * n_embd) 0.0 in
    let dk_rot = Array.make (seq_len * n_embd) 0.0 in

    (* Temporary buffers — reused across heads *)
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

      (* Extract dAttn_h and V_h *)
      extract_head out_grad dout_h h seq_len;
      extract_head v.data v_h h seq_len;

      (* dWeights[S,S] = dAttn_h[S,d] @ V_h[S,d]^T — BLAS *)
      blas_dgemm 2 seq_len seq_len head_dim dout_h v_h dw;

      (* dV_h[S,d] = Weights^T[S,S] @ dAttn_h[S,d] — BLAS *)
      blas_dgemm 4 seq_len head_dim seq_len wt dout_h dv_h;
      insert_head_accum dv_h v.grad h seq_len;

      (* Softmax backward: dscores[i,j] = wt[i,j] * (dw[i,j] - dot_i) * scale
         Reuse dw array for dscores in-place. *)
      for i = 0 to seq_len - 1 do
        let dot = ref 0.0 in
        for j = 0 to i do
          dot := !dot +. dw.(i * seq_len + j) *. wt.(i * seq_len + j)
        done;
        for j = 0 to i do
          dw.(i * seq_len + j) <-
            wt.(i * seq_len + j) *. (dw.(i * seq_len + j) -. !dot) *. scale
        done;
        (* Zero upper triangle — dscores[i,j]=0 for j>i since wt[i,j]=0 *)
        for j = i + 1 to seq_len - 1 do dw.(i * seq_len + j) <- 0.0 done
      done;
      (* dw is now dScores *)

      (* dQ_rot_h[S,d] = dScores[S,S] @ K_rot_h[S,d] — BLAS *)
      extract_head k_rot k_h h seq_len;
      blas_dgemm 0 seq_len head_dim seq_len dw k_h dq_h;
      insert_head_accum dq_h dq_rot h seq_len;

      (* dK_rot_h[S,d] = dScores^T[S,S] @ Q_rot_h[S,d] — BLAS *)
      extract_head q_rot q_h h seq_len;
      blas_dgemm 4 seq_len head_dim seq_len dw q_h dk_h;
      insert_head_accum dk_h dk_rot h seq_len
    done;

    (* Inverse RoPE: rotate gradients back to pre-rotation space *)
    for pos = 0 to seq_len - 1 do
      let cos_p = rope_cos.(pos) and sin_p = rope_sin.(pos) in
      let row = pos * n_embd in
      for h = 0 to n_head - 1 do
        let base = row + h * head_dim in
        for i = 0 to half_dim - 1 do
          let j0 = base + 2 * i and j1 = base + 2 * i + 1 in
          let c = cos_p.(i) and s = sin_p.(i) in
          q.grad.(j0) <- q.grad.(j0) +. dq_rot.(j0) *. c +. dq_rot.(j1) *. s;
          q.grad.(j1) <- q.grad.(j1) -. dq_rot.(j0) *. s +. dq_rot.(j1) *. c;
          k.grad.(j0) <- k.grad.(j0) +. dk_rot.(j0) *. c +. dk_rot.(j1) *. s;
          k.grad.(j1) <- k.grad.(j1) -. dk_rot.(j0) *. s +. dk_rot.(j1) *. c
        done
      done
    done
  in
  { id = fresh_id (); data = out_data; grad = out_grad;
    shape = [|seq_len; n_embd|]; children = [|q; k; v|];
    backward_fn = backward }

(* Batched transformer block *)
let batch_transformer_block x layer seq_len =
  let xn = batch_rmsnorm ~dim:n_embd x in
  let q = batch_matmul layer.attn_wq xn in
  let k = batch_matmul layer.attn_wk xn in
  let v = batch_matmul layer.attn_wv xn in
  let attn_out = batch_fused_attention q k v seq_len in
  let attn_proj = batch_matmul layer.attn_wo attn_out in
  let x = tensor_add x attn_proj in
  let xn = batch_rmsnorm ~dim:n_embd x in
  let mlp_h = batch_matmul layer.mlp_fc1 xn |> tensor_gelu in
  let mlp_out = batch_matmul layer.mlp_fc2 mlp_h in
  tensor_add x mlp_out

let gpt_forward_batch model tokens seq_len =
  let x = batch_embed model.wte tokens n_embd in
  let x = batch_rmsnorm ~dim:n_embd x in
  let x = Array.fold_left (fun x layer ->
    batch_transformer_block x layer seq_len
  ) x model.layers in
  batch_matmul model.lm_head x

(* ══════════════════════════════════════════════════════════════════════
   INFERENCE — single-token with KV cache (unchanged from Stage 6/7)
   ══════════════════════════════════════════════════════════════════════ *)

let embed_token model token_id =
  tensor_row model.wte token_id |> tensor_rmsnorm

let fused_multi_head_attention x layer kv =
  let q_raw = matmul layer.attn_wq x in
  let k_raw = matmul layer.attn_wk x in
  let v = matmul layer.attn_wv x in
  let pos = kv.len in
  kv.k_nodes.(pos) <- k_raw;
  kv.v_nodes.(pos) <- v;
  let q_rot = Array.create_float n_embd in
  let cos_p = rope_cos.(pos) and sin_p = rope_sin.(pos) in
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
  let attn_out = { id = fresh_id (); data = out_data; grad = out_grad;
    shape = [|n_embd|]; children = [||]; backward_fn = backward } in
  matmul layer.attn_wo attn_out

let mlp_block x layer =
  matmul layer.mlp_fc1 x |> tensor_gelu |> fun h -> matmul layer.mlp_fc2 h

let transformer_block x layer kv =
  let x =
    tensor_rmsnorm x |> fun xn ->
    fused_multi_head_attention xn layer kv |> tensor_add x in
  tensor_rmsnorm x |> fun xn -> mlp_block xn layer |> tensor_add x

let gpt_forward model token_id kv_caches =
  let x = embed_token model token_id in
  let x = Array.to_list model.layers
    |> List.mapi (fun li layer -> (li, layer))
    |> List.fold_left (fun x (li, layer) ->
      transformer_block x layer kv_caches.(li)) x in
  matmul model.lm_head x

(* ══════════════════════════════════════════════════════════════════════
   TRAINING (uses BPE encode instead of char tokenize)
   ══════════════════════════════════════════════════════════════════════ *)

type adam_state = { m : float array; v : float array }
(* Higher peak LR — the cosine schedule now stretches over 20K steps,
   so at step 10K (midpoint) the LR is still ~0.0015 instead of near-zero.
   Gradient clipping at norm 1.0 prevents the higher LR from exploding. *)
let learning_rate = 0.003
let beta1 = 0.85
let beta2 = 0.99
let eps_adam = 1e-8
let max_grad_norm = 1.0
let warmup_steps = 400

let init_adam params =
  let total = Array.fold_left (fun acc p -> acc + Array.length p.data) 0 params in
  { m = Array.make total 0.0; v = Array.make total 0.0 }

let get_lr step num_steps =
  if step < warmup_steps then
    learning_rate *. float_of_int step /. float_of_int warmup_steps
  else
    let progress =
      float_of_int (step - warmup_steps)
      /. float_of_int (num_steps - warmup_steps) in
    learning_rate *. 0.5 *. (1.0 +. cos (Float.pi *. progress))

let clip_grad_norm params =
  let norm_sq = ref 0.0 in
  Array.iter (fun p ->
    for i = 0 to Array.length p.grad - 1 do
      norm_sq := !norm_sq +. p.grad.(i) *. p.grad.(i)
    done
  ) params;
  let norm = sqrt !norm_sq in
  if norm > max_grad_norm then begin
    let scale = max_grad_norm /. norm in
    Array.iter (fun p ->
      for i = 0 to Array.length p.grad - 1 do
        p.grad.(i) <- p.grad.(i) *. scale
      done
    ) params
  end

let adam_step params adam step num_steps =
  let lr_t = get_lr step num_steps in
  let bc1 = 1.0 /. (1.0 -. beta1 ** float_of_int (step + 1)) in
  let bc2 = 1.0 /. (1.0 -. beta2 ** float_of_int (step + 1)) in
  let offset = ref 0 in
  params |> Array.iter (fun p ->
    let n = Array.length p.data in
    for i = 0 to n - 1 do
      let fi = !offset + i in
      adam.m.(fi) <- beta1 *. adam.m.(fi) +. (1.0 -. beta1) *. p.grad.(i);
      adam.v.(fi) <- beta2 *. adam.v.(fi) +. (1.0 -. beta2) *. p.grad.(i) *. p.grad.(i);
      let m_hat = adam.m.(fi) *. bc1 in
      let v_hat = adam.v.(fi) *. bc2 in
      p.data.(i) <- p.data.(i) -. lr_t *. m_hat /. (sqrt v_hat +. eps_adam);
      p.grad.(i) <- 0.0
    done;
    offset := !offset + n)

let compute_loss model tokens =
  let seq_len = min block_size (Array.length tokens - 1) in
  let input_tokens = Array.sub tokens 0 seq_len in
  let logits = gpt_forward_batch model input_tokens seq_len in
  let losses = Array.init seq_len (fun i ->
    let logits_i = tensor_row logits i in
    let probs_i = tensor_softmax logits_i in
    tensor_nll probs_i tokens.(i + 1)
  ) in
  (tensor_mean losses, seq_len)

(* Pre-tokenize all docs at startup. BPE encoding is O(n_merges * doc_len)
   per doc — doing it once saves ~500 hash lookups × doc_len per training
   step. With 16K docs this takes ~1-2s but eliminates all per-step
   tokenization overhead. *)
let pre_tokenize docs merges char_to_id bos_id =
  let t0 = Sys.time () in
  let tokenized = Array.map (fun doc ->
    bpe_encode merges char_to_id bos_id doc
  ) docs in
  Printf.printf "pre-tokenized %d docs in %.2fs\n%!"
    (Array.length docs) (Sys.time () -. t0);
  tokenized

let train model params tokenized_docs num_steps =
  let adam = init_adam params in
  (* RUNNING AVERAGE LOSS: accumulate loss over 500-step windows.
     Per-doc loss is too noisy (bounces 2.4-4.6 depending on doc
     difficulty). Averaging over 500 steps shows the real trend. *)
  let loss_sum = ref 0.0 in
  for step = 0 to num_steps - 1 do
    let tokens = tokenized_docs.(step mod Array.length tokenized_docs) in
    let (loss, _) = compute_loss model tokens in
    loss_sum := !loss_sum +. loss.data.(0);
    backward loss;
    clip_grad_norm params;
    adam_step params adam step num_steps;
    if (step + 1) mod 2500 = 0 then begin
      Printf.printf "step %5d / %5d | loss %.4f\n%!"
        (step + 1) num_steps (!loss_sum /. 2500.0);
      loss_sum := 0.0
    end
  done

(* ══════════════════════════════════════════════════════════════════════
   CHECKPOINT SAVE / LOAD
   ══════════════════════════════════════════════════════════════════════

   Save all parameter data arrays to a binary file using Marshal.
   On load, validate that the parameter count and sizes match, then
   blit the saved data into the live parameter tensors.

   Format: Marshal'd float array array (one entry per param tensor).
   Deterministic — the param order in collect_params is fixed. *)

let save_checkpoint filename params =
  let oc = open_out_bin filename in
  let data_arrays = Array.map (fun p -> p.data) params in
  Marshal.to_channel oc data_arrays [Marshal.No_sharing];
  close_out oc;
  let total = Array.fold_left (fun acc p -> acc + Array.length p.data) 0 params in
  Printf.printf "saved checkpoint to %s (%d params)\n%!" filename total

let load_checkpoint filename params =
  let ic = open_in_bin filename in
  let data_arrays : float array array = Marshal.from_channel ic in
  close_in ic;
  if Array.length data_arrays <> Array.length params then
    failwith (Printf.sprintf "checkpoint has %d param tensors, model has %d"
      (Array.length data_arrays) (Array.length params));
  Array.iteri (fun i saved ->
    if Array.length saved <> Array.length params.(i).data then
      failwith (Printf.sprintf "param %d: checkpoint has %d elements, model has %d"
        i (Array.length saved) (Array.length params.(i).data));
    Array.blit saved 0 params.(i).data 0 (Array.length saved)
  ) data_arrays;
  let total = Array.fold_left (fun acc a -> acc + Array.length a) 0 data_arrays in
  Printf.printf "loaded checkpoint from %s (%d params)\n%!" filename total

(* ══════════════════════════════════════════════════════════════════════
   INFERENCE + MAIN (updated for BPE decode)
   ══════════════════════════════════════════════════════════════════════ *)

(* Generate text using the BPE vocabulary.
   Decode: vocab.(token_id) gives the token's surface string (may be
   multiple characters for merged tokens, e.g. "the", "ing", " of"). *)
let generate_sample model vocab bos_id temperature =
  let kv_caches = Array.init n_layer (fun _ -> make_kv_cache ()) in
  let token_id = ref bos_id in
  let buf = Buffer.create 256 in
  let pos = ref 0 in
  while !pos < block_size do
    let logits = gpt_forward model !token_id kv_caches in
    let probs = tensor_scale logits (1.0 /. temperature) |> tensor_softmax in
    token_id := weighted_choice probs.data;
    if !token_id = bos_id then pos := block_size
    else begin Buffer.add_string buf vocab.(!token_id); incr pos end
  done;
  Buffer.contents buf

(* Prompted generation: BPE-encode the prompt, feed each token through
   the model to fill the KV cache (no sampling during prompt), then
   sample continuation tokens autoregressively. *)
let generate_prompted model vocab merges char_to_id bos_id prompt temperature =
  let kv_caches = Array.init n_layer (fun _ -> make_kv_cache ()) in
  let prompt_ids = bpe_encode merges char_to_id bos_id prompt in
  let buf = Buffer.create 256 in
  (* Feed prompt tokens through the model (prefill) *)
  let n_prompt = Array.length prompt_ids in
  for i = 0 to n_prompt - 2 do
    ignore (gpt_forward model prompt_ids.(i) kv_caches);
    (* Decode prompt tokens (skip BOS at position 0) *)
    if i > 0 then Buffer.add_string buf vocab.(prompt_ids.(i))
  done;
  (* Sample from last prompt token onward *)
  let token_id = ref prompt_ids.(n_prompt - 1) in
  if !token_id <> bos_id then Buffer.add_string buf vocab.(!token_id);
  let pos = ref n_prompt in
  while !pos < block_size do
    let logits = gpt_forward model !token_id kv_caches in
    let probs = tensor_scale logits (1.0 /. temperature) |> tensor_softmax in
    token_id := weighted_choice probs.data;
    if !token_id = bos_id then pos := block_size
    else begin Buffer.add_string buf vocab.(!token_id); incr pos end
  done;
  Buffer.contents buf

let () =
  let t_start = Sys.time () in
  let checkpoint_file = "microgpt_tuned.bin" in

  (* BPE is always needed (fast, ~3s) — deterministic from corpus *)
  let docs = load_docs "input.txt" in
  Printf.printf "num docs: %d\n" (Array.length docs);
  let (vocab, merges, char_to_id, bos_id, vocab_size) =
    bpe_train docs n_merges in

  (* Init model, then either load checkpoint or train from scratch *)
  let model = init_model vocab_size in
  let params = collect_params model in
  let total_params =
    Array.fold_left (fun acc p -> acc + Array.length p.data) 0 params in
  Printf.printf "num params: %d\n" total_params;

  if Array.length Sys.argv > 1 && Sys.argv.(1) = "--load" then begin
    (* Load saved weights — skip training entirely *)
    load_checkpoint checkpoint_file params;
    Printf.printf "skipping training (loaded from %s)\n%!" checkpoint_file
  end else begin
    (* Train from scratch and save *)
    let tokenized_docs = pre_tokenize docs merges char_to_id bos_id in
    train model params tokenized_docs 100000;
    save_checkpoint checkpoint_file params
  end;

  (* Check for --prompt "text" in argv *)
  let prompt_text = ref "" in
  for i = 1 to Array.length Sys.argv - 2 do
    if Sys.argv.(i) = "--prompt" then prompt_text := Sys.argv.(i + 1)
  done;

  if !prompt_text <> "" then begin
    Printf.printf "--- prompted generation ---\n";
    Printf.printf "prompt: %s\n" !prompt_text;
    for i = 1 to 10 do
      generate_prompted model vocab merges char_to_id bos_id !prompt_text 0.5
      |> Printf.printf "  %2d: %s\n" i
    done
  end else begin
    Printf.printf "--- inference (new, hallucinated text) ---\n";
    for i = 1 to 20 do
      generate_sample model vocab bos_id 0.5
      |> Printf.printf "sample %2d: %s\n" i
    done
  end;
  Printf.printf "total time: %.2fs\n" (Sys.time () -. t_start)
