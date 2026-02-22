(* utils.ml — Shared utility functions
   =====================================

   Small helpers used across multiple modules:
   - is_word_boundary / strip_word: word boundary detection and stripping
   - random_gauss: Box-Muller normal distribution sampling
   - shuffle: Fisher-Yates in-place array shuffle
   - weighted_choice: sample from a probability distribution
   - load_docs: read a text file into an array of non-empty lines *)

(* Characters treated as word boundaries.
   Used by symbolic.ml (word validation) and knowledge.ml
   (concept extraction) to ensure consistent word segmentation. *)
let is_word_boundary ch =
  ch = ' ' || ch = '\n' || ch = '\t' || ch = '\r'
  || ch = ',' || ch = '.' || ch = ';' || ch = ':'
  || ch = '!' || ch = '?' || ch = '"' || ch = '\''
  || ch = '(' || ch = ')' || ch = '[' || ch = ']'
  || ch = '-' || ch = '/' || ch = '\\'

(* Strip leading and trailing punctuation/whitespace from a word. *)
let strip_word w =
  let n = String.length w in
  let i = ref 0 in
  while !i < n && is_word_boundary w.[!i] do incr i done;
  let j = ref (n - 1) in
  while !j >= !i && is_word_boundary w.[!j] do decr j done;
  if !i > !j then "" else String.sub w !i (!j - !i + 1)

(* Box-Muller transform: generate a sample from N(mean, std²).
   Rejection-samples until we get a valid unit disk point, then
   transforms to normal distribution. Typically accepts on first try
   (π/4 ≈ 78.5% acceptance rate). *)
let random_gauss ?(mean = 0.0) ?(std = 1.0) () =
  let rec sample () =
    let u = Random.float 2.0 -. 1.0 in
    let v = Random.float 2.0 -. 1.0 in
    let s = u *. u +. v *. v in
    if s >= 1.0 || s = 0.0 then sample ()
    else mean +. std *. u *. sqrt (-2.0 *. log s /. s)
  in
  sample ()

(* Fisher-Yates shuffle — mutates the array in place. *)
let shuffle arr =
  for i = Array.length arr - 1 downto 1 do
    let j = Random.int (i + 1) in
    let tmp = arr.(i) in
    arr.(i) <- arr.(j);
    arr.(j) <- tmp
  done

(* Sample an index from a probability distribution (float array).
   Walks through weights, subtracting from a random threshold until
   it crosses zero. Returns the index of the chosen element. *)
let weighted_choice weights =
  let total = Array.fold_left (+.) 0.0 weights in
  let r = Random.float total in
  Array.fold_left (fun (chosen, remaining) w ->
    if remaining <= 0.0 then (chosen, remaining)
    else if remaining -. w <= 0.0 then (chosen, remaining -. w)
    else (chosen + 1, remaining -. w)
  ) (0, r) weights
  |> fst

(* Top-k sampling: keep only the k highest logits, set the rest to -inf.
   Returns a new array. The original is not modified. *)
let top_k k logits =
  let n = Array.length logits in
  let k = min k n in
  (* Build (value, index) pairs and sort descending by value *)
  let indexed = Array.init n (fun i -> (logits.(i), i)) in
  Array.sort (fun (a, _) (b, _) -> compare b a) indexed;
  (* Find the threshold: the k-th highest value *)
  let threshold = fst indexed.(k - 1) in
  (* Zero out everything below the threshold *)
  Array.init n (fun i ->
    if logits.(i) >= threshold then logits.(i) else neg_infinity
  )

(* Load a text file as an array of non-empty trimmed lines, shuffled.
   Used to load the training corpus (one doc per line). *)
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
