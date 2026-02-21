(* knowledge.ml — Bootstrap Forth dictionary from corpus
   =====================================================

   Extracts concepts from the training corpus and builds the initial
   Forth dictionary. Runs once at startup (like Symbolic.build).

   Pipeline:
   1. Extract concepts: frequent words (freq >= min_freq, len >= min_len)
   2. Build co-occurrence matrix: which concepts appear in the same doc
   3. Build associations: top-N co-occurring concepts per concept
   4. Map BPE tokens to concepts: which tokens activate which concepts
   5. Populate Forth dictionary with Concept entries

   The result is a structured knowledge base that the symbolic layer
   uses to bias generation toward topically coherent text. *)

(* ── Configuration ──────────────────────────────────────────────── *)

(* Minimum word frequency in the corpus to qualify as a concept.
   Too low = noise, too high = misses useful concepts. *)
let min_freq = 20

(* Minimum word length to qualify as a concept.
   Skips "the", "and", "is" etc. which carry no topical signal. *)
let min_len = 4

(* Maximum number of associations per concept.
   Top-N by co-occurrence count. More = broader but noisier signal. *)
let max_associations = 8

(* ── Types ──────────────────────────────────────────────────────── *)

type t = {
  dict : Forth.dictionary;
  token_to_concepts : (int, string list) Hashtbl.t;
  concept_to_tokens : (string, int list) Hashtbl.t;
}

(* ── Step 1: Extract concepts ───────────────────────────────────── *)

(* Count word frequencies across all docs.
   Reuses Utils.strip_word for consistent word boundary handling. *)
let extract_concepts docs =
  let freq = Hashtbl.create 4096 in
  Array.iter (fun doc ->
    let words = String.split_on_char ' ' doc in
    List.iter (fun raw ->
      let w = Utils.strip_word (String.lowercase_ascii raw) in
      if String.length w >= min_len then begin
        let count = try Hashtbl.find freq w with Not_found -> 0 in
        Hashtbl.replace freq w (count + 1)
      end
    ) words
  ) docs;
  (* Filter by minimum frequency *)
  let concepts = Hashtbl.create 512 in
  Hashtbl.iter (fun word count ->
    if count >= min_freq then
      Hashtbl.replace concepts word count
  ) freq;
  concepts

(* ── Step 2: Co-occurrence matrix ───────────────────────────────── *)

(* For each doc, find which concepts are present, then increment
   co-occurrence for every pair. Only stores (a, b) where a < b
   to avoid double-counting. *)
let build_cooccurrence docs concepts =
  let cooccur = Hashtbl.create 8192 in
  Array.iter (fun doc ->
    (* Find concepts present in this doc *)
    let present = Hashtbl.create 32 in
    let words = String.split_on_char ' ' doc in
    List.iter (fun raw ->
      let w = Utils.strip_word (String.lowercase_ascii raw) in
      if Hashtbl.mem concepts w then
        Hashtbl.replace present w ()
    ) words;
    (* For each pair of concepts present, increment co-occurrence *)
    let present_list = Hashtbl.fold (fun k () acc -> k :: acc) present [] in
    List.iter (fun a ->
      List.iter (fun b ->
        if a < b then begin
          let key = (a, b) in
          let count = try Hashtbl.find cooccur key with Not_found -> 0 in
          Hashtbl.replace cooccur key (count + 1)
        end
      ) present_list
    ) present_list
  ) docs;
  cooccur

(* ── Step 3: Build associations ─────────────────────────────────── *)

(* For each concept, find its top-N co-occurring concepts.
   Returns a hashtbl: concept_name -> association list.
   Uses a pre-built per-concept index instead of scanning the entire
   co-occurrence table for every concept. *)
let build_associations concepts cooccur =
  (* Build per-concept index: concept -> (other, count) list *)
  let index = Hashtbl.create (Hashtbl.length concepts) in
  Hashtbl.iter (fun (a, b) count ->
    let existing_a = try Hashtbl.find index a with Not_found -> [] in
    Hashtbl.replace index a ((b, count) :: existing_a);
    let existing_b = try Hashtbl.find index b with Not_found -> [] in
    Hashtbl.replace index b ((a, count) :: existing_b)
  ) cooccur;
  (* Now each concept just looks up its own list *)
  let assoc_table = Hashtbl.create 512 in
  Hashtbl.iter (fun concept _count ->
    let pairs = try Hashtbl.find index concept with Not_found -> [] in
    let sorted = List.sort (fun (_, c1) (_, c2) -> compare c2 c1) pairs in
    let top_n = List.filteri (fun i _ -> i < max_associations) sorted in
    let assoc_weighted = List.map (fun (name, _count) -> (name, 2.0)) top_n in
    Hashtbl.replace assoc_table concept assoc_weighted
  ) concepts;
  assoc_table

(* ── Step 4: Token-concept mapping ──────────────────────────────── *)

(* Map BPE tokens to concepts and vice versa.
   A token maps to a concept if the token string (lowercased, stripped)
   matches the concept name exactly, or if the concept name starts with
   the token string (for multi-token concepts). *)
let build_token_maps vocab concepts =
  let token_to_concepts = Hashtbl.create 1024 in
  let concept_to_tokens = Hashtbl.create 512 in
  let n_vocab = Array.length vocab in
  for tok_id = 0 to n_vocab - 1 do
    let tok_str = String.lowercase_ascii (Utils.strip_word vocab.(tok_id)) in
    if String.length tok_str >= 2 then begin
      (* Check exact match first *)
      if Hashtbl.mem concepts tok_str then begin
        let existing = try Hashtbl.find token_to_concepts tok_id
          with Not_found -> [] in
        Hashtbl.replace token_to_concepts tok_id (tok_str :: existing);
        let existing = try Hashtbl.find concept_to_tokens tok_str
          with Not_found -> [] in
        Hashtbl.replace concept_to_tokens tok_str (tok_id :: existing)
      end;
      (* Check if any concept starts with this token (substring match) *)
      if String.length tok_str >= 3 then
        Hashtbl.iter (fun concept _count ->
          if concept <> tok_str
             && String.length concept > String.length tok_str
             && String.sub concept 0 (String.length tok_str) = tok_str then begin
            let existing = try Hashtbl.find token_to_concepts tok_id
              with Not_found -> [] in
            if not (List.mem concept existing) then
              Hashtbl.replace token_to_concepts tok_id (concept :: existing);
            let existing = try Hashtbl.find concept_to_tokens concept
              with Not_found -> [] in
            if not (List.mem tok_id existing) then
              Hashtbl.replace concept_to_tokens concept (tok_id :: existing)
          end
        ) concepts
    end
  done;
  (token_to_concepts, concept_to_tokens)

(* ── Step 5: Build — the main entry point ───────────────────────── *)

(* Build the complete Forth-based knowledge from the corpus.
   Creates a dictionary with primitives + concept entries,
   plus bidirectional token-concept mappings.

   vocab: BPE vocabulary (string array, token_id -> string)
   docs: training corpus (string array, one doc per line)
   bos_id: beginning-of-sequence token ID (unused here, reserved) *)
let build vocab docs _bos_id =
  (* Step 1: Extract concepts *)
  let concepts = extract_concepts docs in

  (* Step 2: Co-occurrence *)
  let cooccur = build_cooccurrence docs concepts in

  (* Step 3: Associations *)
  let assoc_table = build_associations concepts cooccur in

  (* Step 4: Token-concept mapping *)
  let (token_to_concepts, concept_to_tokens) =
    build_token_maps vocab concepts in

  (* Step 5: Populate Forth dictionary *)
  let dict = Forth.create () in

  (* Find max frequency for normalization *)
  let max_freq = Hashtbl.fold (fun _ count mx -> max count mx) concepts 1 in
  let max_freq_f = float_of_int max_freq in

  Hashtbl.iter (fun concept_name count ->
    let associations =
      try Hashtbl.find assoc_table concept_name
      with Not_found -> [] in
    let token_ids =
      try Hashtbl.find concept_to_tokens concept_name
      with Not_found -> [] in
    let strength = float_of_int count /. max_freq_f in
    Forth.define dict concept_name
      (Forth.Concept { associations; strength; token_ids })
      { Forth.consumed = 0; produced = 1 }
  ) concepts;

  let (n_prim, _n_def, n_concept, n_assoc) = Forth.stats dict in
  Printf.printf "forth: %d primitives, %d concepts, %d associations\n%!"
    n_prim n_concept n_assoc;

  { dict; token_to_concepts; concept_to_tokens }

(* ── TD weight persistence ────────────────────────────────────── *)

(* Save learned association weights to a file.
   Format: Marshal'd list of (concept_name, (assoc_name * weight) list). *)
let save_weights filename dict =
  let data = Hashtbl.fold (fun name entry acc ->
    match entry.Forth.kind with
    | Forth.Concept { associations; _ } -> (name, associations) :: acc
    | _ -> acc
  ) dict.Forth.entries [] in
  let oc = open_out_bin filename in
  Marshal.to_channel oc (data : (string * (string * float) list) list) [];
  close_out oc

(* Load learned association weights from a file.
   Only updates weights for concepts that exist in the current dictionary. *)
let load_weights filename dict =
  if Sys.file_exists filename then begin
    let ic = open_in_bin filename in
    let data : (string * (string * float) list) list = Marshal.from_channel ic in
    close_in ic;
    List.iter (fun (concept_name, weights) ->
      Forth.update_association_weights dict concept_name weights
    ) data;
    Printf.printf "td: loaded weights from %s (%d concepts)\n%!"
      filename (List.length data)
  end
