(* symbolic.ml — Constrained decoding via logit masks/biases
   ===========================================================

   The neural model provides raw logit scores (what's likely).
   The symbolic system masks/biases them to enforce validity
   (what's allowed). No retraining needed — just logit manipulation
   at inference time.

   Two constraints applied in sequence:
   1. Repetition penalty: penalize recently generated tokens
   2. Word validation: mask out tokens that would form invalid words

   Core principle from PLAN.md:
     "the neural model ranks by likelihood,
      the symbolic system defines what's valid."

   Split into two types:
   - knowledge: static data built once from corpus (word sets, vocab)
   - context: per-generation mutable state (ring buffer, partial word) *)

(* Ring buffer size for tracking recent tokens *)
let recent_size = 32

(* Penalty subtracted from logits of recently seen tokens *)
let rep_penalty = 1.5

(* Characters treated as word boundaries when tracking partial words.
   Punctuation attached to words in the corpus (e.g. "soul," "itself.")
   gets stripped during knowledge building, and also resets the partial
   word tracker during generation. *)
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

(* ── Static knowledge (built once from corpus) ───────────────────── *)

type knowledge = {
  valid_words : (string, unit) Hashtbl.t;
  valid_prefixes : (string, unit) Hashtbl.t;
  vocab : string array;
  bos_id : int;
}

(* build: Extract all words from the corpus, strip punctuation,
   lowercase, and build valid_words + valid_prefixes sets.
   Called once at startup — the sets are shared across all generations. *)
let build vocab docs bos_id =
  let valid_words = Hashtbl.create 4096 in
  let valid_prefixes = Hashtbl.create 16384 in
  Array.iter (fun doc ->
    (* Split on whitespace *)
    let words = String.split_on_char ' ' doc in
    List.iter (fun raw ->
      let w = strip_word (String.lowercase_ascii raw) in
      if String.length w > 0 then begin
        Hashtbl.replace valid_words w ();
        for len = 1 to String.length w do
          Hashtbl.replace valid_prefixes (String.sub w 0 len) ()
        done
      end
    ) words
  ) docs;
  Printf.printf "symbolic: %d valid words, %d prefixes\n%!"
    (Hashtbl.length valid_words) (Hashtbl.length valid_prefixes);
  { valid_words; valid_prefixes; vocab; bos_id }

(* ── Per-generation mutable context ───────────────────────────────── *)

type context = {
  know : knowledge;
  recent_tokens : int array;
  mutable pos : int;
  mutable partial_word : string;
}

(* create: Fresh context for one generation run.
   Reuses the static knowledge; only allocates the ring buffer. *)
let create know =
  { know;
    recent_tokens = Array.make recent_size (-1);
    pos = 0;
    partial_word = "";
  }

(* record_token: Add a generated token to the ring buffer and update
   the partial word tracker. Call this after sampling each token.
   Punctuation and whitespace reset the partial word — matching the
   stripping done during knowledge building. *)
let record_token ctx token_id =
  ctx.recent_tokens.(ctx.pos) <- token_id;
  ctx.pos <- (ctx.pos + 1) mod recent_size;
  if token_id >= 0 && token_id < Array.length ctx.know.vocab then begin
    let tok_str = ctx.know.vocab.(token_id) in
    String.iter (fun ch ->
      if is_word_boundary ch then
        ctx.partial_word <- ""
      else
        ctx.partial_word <- ctx.partial_word ^ String.make 1 ch
    ) tok_str
  end

(* ── Constraint 1: Repetition penalty ────────────────────────────── *)

let repetition_penalty logits recent_tokens penalty =
  let biased = Array.copy logits in
  Array.iter (fun tok ->
    if tok >= 0 && tok < Array.length biased then
      biased.(tok) <- biased.(tok) -. penalty
  ) recent_tokens;
  biased

(* ── Constraint 2: Word validation ────────────────────────────────── *)

(* Extract the "word part" from a candidate string — the portion after
   the last word boundary, lowercased. If the string ends with a boundary,
   also check the word completed before it. *)
let word_validation logits vocab partial_word valid_words valid_prefixes =
  let biased = Array.copy logits in
  let n = Array.length logits in
  for tok_id = 0 to n - 1 do
    if biased.(tok_id) > neg_infinity then begin
      let tok_str = vocab.(tok_id) in
      let candidate = String.lowercase_ascii (partial_word ^ tok_str) in
      let clen = String.length candidate in
      if clen > 0 then begin
        (* Find the last word boundary in the candidate *)
        let last_boundary = ref (-1) in
        String.iteri (fun i ch ->
          if is_word_boundary ch then last_boundary := i
        ) candidate;
        if !last_boundary >= 0 then begin
          (* Token contains or crosses a word boundary.
             Check the completed word (text between previous boundary and this one). *)
          let word_start = ref 0 in
          for i = 0 to !last_boundary - 1 do
            if is_word_boundary candidate.[i] then
              word_start := i + 1
          done;
          let completed_word = String.sub candidate !word_start
            (!last_boundary - !word_start) in
          if String.length completed_word > 0
             && not (Hashtbl.mem valid_words completed_word) then
            biased.(tok_id) <- neg_infinity;
          (* Also check the trailing partial after the last boundary *)
          if biased.(tok_id) > neg_infinity && !last_boundary < clen - 1 then begin
            let trailing = String.sub candidate (!last_boundary + 1)
              (clen - !last_boundary - 1) in
            if String.length trailing > 0
               && not (Hashtbl.mem valid_prefixes trailing)
               && not (Hashtbl.mem valid_words trailing) then
              biased.(tok_id) <- neg_infinity
          end
        end else begin
          (* No boundary — entire candidate is a partial word.
             Must be a valid prefix or complete word. *)
          if not (Hashtbl.mem valid_prefixes candidate)
             && not (Hashtbl.mem valid_words candidate) then
            biased.(tok_id) <- neg_infinity
        end
      end
    end
  done;
  biased

(* apply: Apply all symbolic constraints to logits before sampling.
   Returns biased logits. Does NOT modify the original array. *)
let apply ctx logits =
  let biased = repetition_penalty logits ctx.recent_tokens rep_penalty in
  let biased = word_validation biased ctx.know.vocab ctx.partial_word
    ctx.know.valid_words ctx.know.valid_prefixes in
  (* Safety: if all logits are neg_infinity, fall back to just rep penalty
     to avoid a degenerate softmax. This can happen if the model is
     mid-word and no valid continuation exists in the vocabulary. *)
  let all_neg_inf = Array.for_all (fun x -> x = neg_infinity) biased in
  if all_neg_inf then
    repetition_penalty logits ctx.recent_tokens rep_penalty
  else
    biased
