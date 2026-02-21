(* symbolic.ml — Constrained decoding via logit masks/biases
   ===========================================================

   The neural model provides raw logit scores (what's likely).
   The symbolic system masks/biases them to enforce validity
   (what's allowed). No retraining needed — just logit manipulation
   at inference time.

   Five constraints applied in sequence:
   1. Repetition penalty: penalize recently generated tokens
   2. Word boundary: when a complete word is formed, require a boundary next
   3. Word validation: mask out tokens that would form invalid words
   4. Concept coherence: boost tokens associated with active concepts
   5. Topic depth: penalize over-concentration on a single topic

   Core principle:
     "the neural model ranks by likelihood,
      the symbolic system defines what's valid."

   Split into two types:
   - knowledge: static data built once from corpus (word sets, vocab)
   - context: per-generation mutable state (ring buffer, partial word, concepts) *)

(* Ring buffer size for tracking recent tokens *)
let recent_size = 32

(* Penalty subtracted from logits of recently seen tokens *)
let rep_penalty = 1.5

let is_word_boundary = Utils.is_word_boundary
let strip_word = Utils.strip_word

(* ── Static knowledge (built once from corpus) ───────────────────── *)

type knowledge = {
  valid_words : (string, unit) Hashtbl.t;
  valid_prefixes : (string, unit) Hashtbl.t;
  vocab : string array;
  bos_id : int;
  special_ids : int list;
}

let build vocab docs bos_id ?(special_ids = []) () =
  let valid_words = Hashtbl.create 4096 in
  let valid_prefixes = Hashtbl.create 16384 in
  Array.iter (fun doc ->
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
  { valid_words; valid_prefixes; vocab; bos_id; special_ids }

(* ── TD learning types ─────────────────────────────────────────────── *)

type contribution = {
  concept_name : string;
  assoc_name : string;
  weight : float;
  decay : float;
  token_ids : int list;
}

(* ── Per-generation mutable context ───────────────────────────────── *)

type context = {
  know : knowledge;
  concept_know : Knowledge.t option;
  recent_tokens : int array;
  mutable pos : int;
  mutable partial_word : string;
  mutable active_concepts : (string * int) list;
  topic_counts : (string, int) Hashtbl.t;
  mutable contributions : contribution list;
  mutable td_baseline : float;
}

let create ?concept_know know =
  { know;
    concept_know;
    recent_tokens = Array.make recent_size (-1);
    pos = 0;
    partial_word = "";
    active_concepts = [];
    topic_counts = Hashtbl.create 32;
    contributions = [];
    td_baseline = 0.0;
  }

let record_token ctx token_id =
  ctx.recent_tokens.(ctx.pos) <- token_id;
  ctx.pos <- (ctx.pos + 1) mod recent_size;
  if List.mem token_id ctx.know.special_ids then
    ctx.partial_word <- ""
  else if token_id >= 0 && token_id < Array.length ctx.know.vocab then begin
    let tok_str = ctx.know.vocab.(token_id) in
    String.iter (fun ch ->
      if is_word_boundary ch then
        ctx.partial_word <- ""
      else
        ctx.partial_word <- ctx.partial_word ^ String.make 1 ch
    ) tok_str
  end;
  match ctx.concept_know with
  | None -> ()
  | Some knowledge ->
    ctx.active_concepts <-
      List.filter_map (fun (c, age) ->
        if age < 16 then Some (c, age + 1) else None
      ) ctx.active_concepts;
    (match Hashtbl.find_opt knowledge.Knowledge.token_to_concepts token_id with
     | None -> ()
     | Some concepts ->
       List.iter (fun c ->
         ctx.active_concepts <- (c, 0) :: ctx.active_concepts;
         let count =
           try Hashtbl.find ctx.topic_counts c with Not_found -> 0 in
         Hashtbl.replace ctx.topic_counts c (count + 1)
       ) concepts)

(* ── Constraint 1: Repetition penalty ────────────────────────────── *)

let repetition_penalty logits recent_tokens penalty =
  let biased = Array.copy logits in
  Array.iter (fun tok ->
    if tok >= 0 && tok < Array.length biased then
      biased.(tok) <- biased.(tok) -. penalty
  ) recent_tokens;
  biased

(* ── Constraint 2: Word boundary ──────────────────────────────────── *)

let boundary_penalty = 5.0

let word_boundary_bias logits vocab partial_word valid_words =
  if String.length partial_word = 0 then logits
  else if not (Hashtbl.mem valid_words (String.lowercase_ascii partial_word)) then logits
  else begin
    let biased = Array.copy logits in
    let n = Array.length logits in
    for tok_id = 0 to n - 1 do
      if biased.(tok_id) > neg_infinity then begin
        let tok_str = vocab.(tok_id) in
        if String.length tok_str > 0
           && not (is_word_boundary tok_str.[0]) then
          biased.(tok_id) <- biased.(tok_id) -. boundary_penalty
      end
    done;
    biased
  end

(* ── Constraint 3: Word validation ────────────────────────────────── *)

let word_validation logits vocab partial_word valid_words valid_prefixes =
  let biased = Array.copy logits in
  let n = Array.length logits in
  for tok_id = 0 to n - 1 do
    if biased.(tok_id) > neg_infinity then begin
      let tok_str = vocab.(tok_id) in
      let candidate = String.lowercase_ascii (partial_word ^ tok_str) in
      let clen = String.length candidate in
      if clen > 0 then begin
        let last_boundary = ref (-1) in
        String.iteri (fun i ch ->
          if is_word_boundary ch then last_boundary := i
        ) candidate;
        if !last_boundary >= 0 then begin
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
          if biased.(tok_id) > neg_infinity && !last_boundary < clen - 1 then begin
            let trailing = String.sub candidate (!last_boundary + 1)
              (clen - !last_boundary - 1) in
            if String.length trailing > 0
               && not (Hashtbl.mem valid_prefixes trailing)
               && not (Hashtbl.mem valid_words trailing) then
              biased.(tok_id) <- neg_infinity
          end
        end else begin
          if not (Hashtbl.mem valid_prefixes candidate)
             && not (Hashtbl.mem valid_words candidate) then
            biased.(tok_id) <- neg_infinity
        end
      end
    end
  done;
  biased

(* ── Constraint 4: Concept coherence ─────────────────────────────── *)

let coherence_decay = 0.85

let concept_coherence ctx logits active_concepts knowledge =
  if active_concepts = [] then logits
  else begin
    let biased = Array.copy logits in
    List.iter (fun (concept_name, age) ->
      let decay = coherence_decay ** float_of_int age in
      let assocs = Knowledge.associations knowledge concept_name in
      List.iter (fun (assoc_name, weight) ->
        let boost = weight *. decay in
        if boost > 0.01 then begin
          match Hashtbl.find_opt knowledge.Knowledge.concept_to_tokens assoc_name with
          | None -> ()
          | Some token_ids ->
            List.iter (fun tid ->
              if tid >= 0 && tid < Array.length biased then
                biased.(tid) <- biased.(tid) +. boost
            ) token_ids;
            ctx.contributions <- {
              concept_name; assoc_name; weight; decay; token_ids;
            } :: ctx.contributions
        end
      ) assocs
    ) active_concepts;
    biased
  end

(* ── Constraint 5: Topic depth penalty ──────────────────────────── *)

let max_topic_depth = 4
let depth_penalty = 0.5

let topic_depth_penalty logits topic_counts knowledge =
  let any_excess = Hashtbl.fold (fun _ count found ->
    found || count > max_topic_depth
  ) topic_counts false in
  if not any_excess then logits
  else begin
    let biased = Array.copy logits in
    Hashtbl.iter (fun concept_name count ->
      if count > max_topic_depth then begin
        let excess = count - max_topic_depth in
        let penalty = depth_penalty *. float_of_int excess in
        match Hashtbl.find_opt knowledge.Knowledge.concept_to_tokens concept_name with
        | None -> ()
        | Some token_ids ->
          List.iter (fun tid ->
            if tid >= 0 && tid < Array.length biased then
              biased.(tid) <- biased.(tid) -. penalty
          ) token_ids
      end
    ) topic_counts;
    biased
  end

(* ── Apply all constraints ──────────────────────────────────────── *)

let apply ctx logits =
  ctx.contributions <- [];
  let biased = repetition_penalty logits ctx.recent_tokens rep_penalty in
  let special_saved =
    List.map (fun id -> (id, biased.(id))) ctx.know.special_ids in
  let biased = word_boundary_bias biased ctx.know.vocab ctx.partial_word
    ctx.know.valid_words in
  let biased = word_validation biased ctx.know.vocab ctx.partial_word
    ctx.know.valid_words ctx.know.valid_prefixes in
  List.iter (fun (id, v) ->
    if id >= 0 && id < Array.length biased then
      biased.(id) <- v
  ) special_saved;
  let biased = match ctx.concept_know with
    | None -> biased
    | Some knowledge ->
      let biased = concept_coherence ctx biased ctx.active_concepts knowledge in
      topic_depth_penalty biased ctx.topic_counts knowledge
  in
  let all_neg_inf = Array.for_all (fun x -> x = neg_infinity) biased in
  if all_neg_inf then
    repetition_penalty logits ctx.recent_tokens rep_penalty
  else
    biased

(* ── TD learning update ──────────────────────────────────────────── *)

let td_alpha = 0.005
let td_max_weight = 5.0
let td_baseline_decay = 0.99
let td_verbose = try ignore (Sys.getenv "VIDYA_TD_VERBOSE"); true
                 with Not_found -> false

let td_update ctx raw_logits chosen_token =
  match ctx.concept_know with
  | None -> ()
  | Some knowledge ->
    if ctx.contributions = [] then ()
    else begin
      let reward = raw_logits.(chosen_token) in
      let delta = reward -. ctx.td_baseline in
      ctx.td_baseline <-
        td_baseline_decay *. ctx.td_baseline
        +. (1.0 -. td_baseline_decay) *. reward;
      if td_verbose then
        Printf.printf "td: reward=%.3f baseline=%.3f delta=%.3f\n%!"
          reward ctx.td_baseline delta;
      List.iter (fun contrib ->
        let new_weight = contrib.weight +. td_alpha *. delta *. contrib.decay in
        let clamped = Float.max 0.0 (Float.min td_max_weight new_weight) in
        if clamped <> contrib.weight then begin
          Knowledge.update_association_weight knowledge
            contrib.concept_name contrib.assoc_name clamped;
          if td_verbose then
            Printf.printf "  td: %s->%s %.3f->%.3f\n%!"
              contrib.concept_name contrib.assoc_name contrib.weight clamped
        end
      ) ctx.contributions
    end
