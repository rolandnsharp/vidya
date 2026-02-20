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
      the symbolic system defines what's valid." *)

(* Ring buffer size for tracking recent tokens *)
let recent_size = 32

(* Penalty subtracted from logits of recently seen tokens *)
let rep_penalty = 1.5

type context = {
  recent_tokens : int array;           (* ring buffer of last N tokens *)
  mutable pos : int;                   (* write position in ring buffer *)
  valid_words : (string, unit) Hashtbl.t;    (* set of valid whole words *)
  valid_prefixes : (string, unit) Hashtbl.t; (* set of valid word prefixes *)
  vocab : string array;                (* BPE vocab for decoding token strings *)
  bos_id : int;
  mutable partial_word : string;       (* current incomplete word being built *)
}

(* create: Build symbolic context from vocabulary and corpus.
   Extracts all whitespace-delimited words from docs to build the
   valid_words set. Also builds a valid_prefixes set containing
   every prefix of every valid word (for partial matching during
   generation). *)
let create vocab docs bos_id =
  let valid_words = Hashtbl.create 4096 in
  let valid_prefixes = Hashtbl.create 16384 in
  Array.iter (fun doc ->
    let words = String.split_on_char ' ' doc in
    List.iter (fun w ->
      let w = String.lowercase_ascii w in
      if String.length w > 0 then begin
        Hashtbl.replace valid_words w ();
        (* Add all prefixes of this word *)
        for len = 1 to String.length w do
          Hashtbl.replace valid_prefixes (String.sub w 0 len) ()
        done
      end
    ) words
  ) docs;
  Printf.printf "symbolic: %d valid words, %d prefixes\n%!"
    (Hashtbl.length valid_words) (Hashtbl.length valid_prefixes);
  { recent_tokens = Array.make recent_size (-1);
    pos = 0;
    valid_words;
    valid_prefixes;
    vocab;
    bos_id;
    partial_word = "";
  }

(* record_token: Add a generated token to the ring buffer and update
   the partial word tracker. Call this after sampling each token. *)
let record_token ctx token_id =
  ctx.recent_tokens.(ctx.pos) <- token_id;
  ctx.pos <- (ctx.pos + 1) mod recent_size;
  (* Update partial word tracking *)
  if token_id >= 0 && token_id < Array.length ctx.vocab then begin
    let tok_str = ctx.vocab.(token_id) in
    (* Walk through the token string character by character.
       Spaces delimit words — when we hit a space, the partial
       word resets. *)
    String.iter (fun ch ->
      if ch = ' ' || ch = '\n' || ch = '\t' then
        ctx.partial_word <- ""
      else
        ctx.partial_word <- ctx.partial_word ^ String.make 1 ch
    ) tok_str
  end

(* ── Constraint 1: Repetition penalty ────────────────────────────── *)

(* Subtract a fixed penalty from logits of any token that appears
   in the recent_tokens ring buffer. Prevents loops like
   "the the the" and repetitive phrasing. *)
let repetition_penalty logits recent_tokens penalty =
  let biased = Array.copy logits in
  Array.iter (fun tok ->
    if tok >= 0 && tok < Array.length biased then
      biased.(tok) <- biased.(tok) -. penalty
  ) recent_tokens;
  biased

(* ── Constraint 2: Word validation ────────────────────────────────── *)

(* Check if a token would produce a valid word or valid prefix.
   When a BPE token extends the current partial word:
   - If it contains a space, check that the completed word is valid
   - If no space, check that the candidate is a valid prefix

   Tokens that would create invalid words get logit = neg_infinity. *)
let word_validation logits vocab partial_word valid_words valid_prefixes =
  let biased = Array.copy logits in
  let n = Array.length logits in
  for tok_id = 0 to n - 1 do
    if biased.(tok_id) > neg_infinity then begin
      let tok_str = vocab.(tok_id) in
      let candidate = partial_word ^ tok_str in
      let candidate_lower = String.lowercase_ascii candidate in
      (* Find the last space in the candidate to extract the word part *)
      let has_space = ref false in
      let last_space = ref (-1) in
      String.iteri (fun i ch ->
        if ch = ' ' || ch = '\n' || ch = '\t' then begin
          has_space := true;
          last_space := i
        end
      ) candidate;
      if !has_space then begin
        (* Token completes a word (contains a space).
           Check the word formed before the last space. *)
        if !last_space > 0 then begin
          (* Extract completed word: everything before the space boundary.
             For "partial_wordtok" where tok contains a space at position p,
             the completed word is candidate[0..space-1].
             But we need to find the START of the current word too. *)
          let word_start = ref 0 in
          for i = 0 to !last_space - 1 do
            let ch = candidate_lower.[i] in
            if ch = ' ' || ch = '\n' || ch = '\t' then
              word_start := i + 1
          done;
          let completed_word = String.sub candidate_lower !word_start
            (!last_space - !word_start) in
          if String.length completed_word > 0
             && not (Hashtbl.mem valid_words completed_word) then
            biased.(tok_id) <- neg_infinity
        end;
        (* Also check the trailing partial after the last space *)
        let trailing = String.sub candidate_lower (!last_space + 1)
          (String.length candidate_lower - !last_space - 1) in
        if String.length trailing > 0
           && not (Hashtbl.mem valid_prefixes trailing)
           && biased.(tok_id) > neg_infinity then
          biased.(tok_id) <- neg_infinity
      end else begin
        (* No space — entire candidate is a partial word.
           Check it's a valid prefix (could become a valid word). *)
        if String.length candidate_lower > 0
           && not (Hashtbl.mem valid_prefixes candidate_lower)
           && not (Hashtbl.mem valid_words candidate_lower) then
          biased.(tok_id) <- neg_infinity
      end
    end
  done;
  biased

(* apply: Apply all symbolic constraints to logits before sampling.
   Returns biased logits. Does NOT modify the original array. *)
let apply ctx logits =
  let biased = repetition_penalty logits ctx.recent_tokens rep_penalty in
  let biased = word_validation biased ctx.vocab ctx.partial_word
    ctx.valid_words ctx.valid_prefixes in
  (* Safety: if all logits are neg_infinity, fall back to just rep penalty
     to avoid a degenerate softmax. This can happen if the model is
     mid-word and no valid continuation exists in the vocabulary. *)
  let all_neg_inf = Array.for_all (fun x -> x = neg_infinity) biased in
  if all_neg_inf then
    repetition_penalty logits ctx.recent_tokens rep_penalty
  else
    biased
