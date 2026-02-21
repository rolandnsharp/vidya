(* knowledge.ml — Concept knowledge extracted from corpus
   =====================================================

   Extracts concepts from the training corpus and builds a structured
   knowledge base in plain OCaml data structures.

   Pipeline:
   1. Extract concepts: frequent words (freq >= min_freq, len >= min_len)
   2. Build co-occurrence matrix: which concepts appear in the same doc
   3. Build associations: top-N co-occurring concepts per concept
   4. Map BPE tokens to concepts: which tokens activate which concepts

   The result is used by the symbolic layer to bias generation toward
   topically coherent text. *)

(* ── Configuration ──────────────────────────────────────────────── *)

let min_freq = 20
let min_len = 4
let max_associations = 8

(* ── Types ──────────────────────────────────────────────────────── *)

type concept = {
  mutable associations : (string * float) list;
  strength : float;
  token_ids : int list;
}

type t = {
  concepts : (string, concept) Hashtbl.t;
  token_to_concepts : (int, string list) Hashtbl.t;
  concept_to_tokens : (string, int list) Hashtbl.t;
}

(* ── Concept queries ────────────────────────────────────────────── *)

let associations t name =
  match Hashtbl.find_opt t.concepts name with
  | Some c -> c.associations
  | None -> []

let update_association_weight t concept_name assoc_name new_weight =
  match Hashtbl.find_opt t.concepts concept_name with
  | Some c ->
    c.associations <- List.map (fun (n, w) ->
      if n = assoc_name then (n, new_weight) else (n, w)
    ) c.associations
  | None -> ()

let update_association_weights t concept_name updates =
  match Hashtbl.find_opt t.concepts concept_name with
  | Some c ->
    c.associations <- List.map (fun (n, w) ->
      match List.assoc_opt n updates with
      | Some new_w -> (n, new_w)
      | None -> (n, w)
    ) c.associations
  | None -> ()

(* ── Step 1: Extract concepts ───────────────────────────────────── *)

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
  let concepts = Hashtbl.create 512 in
  Hashtbl.iter (fun word count ->
    if count >= min_freq then
      Hashtbl.replace concepts word count
  ) freq;
  concepts

(* ── Step 2: Co-occurrence matrix ───────────────────────────────── *)

let build_cooccurrence docs concepts =
  let cooccur = Hashtbl.create 8192 in
  Array.iter (fun doc ->
    let present = Hashtbl.create 32 in
    let words = String.split_on_char ' ' doc in
    List.iter (fun raw ->
      let w = Utils.strip_word (String.lowercase_ascii raw) in
      if Hashtbl.mem concepts w then
        Hashtbl.replace present w ()
    ) words;
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

let build_associations concepts cooccur =
  let index = Hashtbl.create (Hashtbl.length concepts) in
  Hashtbl.iter (fun (a, b) count ->
    let existing_a = try Hashtbl.find index a with Not_found -> [] in
    Hashtbl.replace index a ((b, count) :: existing_a);
    let existing_b = try Hashtbl.find index b with Not_found -> [] in
    Hashtbl.replace index b ((a, count) :: existing_b)
  ) cooccur;
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

let build_token_maps vocab concepts =
  let token_to_concepts = Hashtbl.create 1024 in
  let concept_to_tokens = Hashtbl.create 512 in
  let n_vocab = Array.length vocab in
  for tok_id = 0 to n_vocab - 1 do
    let tok_str = String.lowercase_ascii (Utils.strip_word vocab.(tok_id)) in
    if String.length tok_str >= 2 then begin
      if Hashtbl.mem concepts tok_str then begin
        let existing = try Hashtbl.find token_to_concepts tok_id
          with Not_found -> [] in
        Hashtbl.replace token_to_concepts tok_id (tok_str :: existing);
        let existing = try Hashtbl.find concept_to_tokens tok_str
          with Not_found -> [] in
        Hashtbl.replace concept_to_tokens tok_str (tok_id :: existing)
      end;
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

(* ── Build — the main entry point ───────────────────────────────── *)

let build vocab docs _bos_id =
  let freq = extract_concepts docs in
  let cooccur = build_cooccurrence docs freq in
  let assoc_table = build_associations freq cooccur in
  let (token_to_concepts, concept_to_tokens) =
    build_token_maps vocab freq in

  let max_freq = Hashtbl.fold (fun _ count mx -> max count mx) freq 1 in
  let max_freq_f = float_of_int max_freq in

  let concepts = Hashtbl.create (Hashtbl.length freq) in
  Hashtbl.iter (fun concept_name count ->
    let associations =
      try Hashtbl.find assoc_table concept_name
      with Not_found -> [] in
    let token_ids =
      try Hashtbl.find concept_to_tokens concept_name
      with Not_found -> [] in
    let strength = float_of_int count /. max_freq_f in
    Hashtbl.replace concepts concept_name { associations; strength; token_ids }
  ) freq;

  let n_concepts = Hashtbl.length concepts in
  let n_assoc = Hashtbl.fold (fun _ c acc ->
    acc + List.length c.associations) concepts 0 in
  Printf.printf "knowledge: %d concepts, %d associations\n%!"
    n_concepts n_assoc;

  { concepts; token_to_concepts; concept_to_tokens }

(* ── TD weight persistence ────────────────────────────────────── *)

let save_weights filename t =
  let data = Hashtbl.fold (fun name concept acc ->
    (name, concept.associations) :: acc
  ) t.concepts [] in
  let oc = open_out_bin filename in
  Marshal.to_channel oc (data : (string * (string * float) list) list) [];
  close_out oc

let load_weights filename t =
  if Sys.file_exists filename then begin
    let ic = open_in_bin filename in
    let data : (string * (string * float) list) list = Marshal.from_channel ic in
    close_in ic;
    List.iter (fun (concept_name, weights) ->
      update_association_weights t concept_name weights
    ) data;
    Printf.printf "td: loaded weights from %s (%d concepts)\n%!"
      filename (List.length data)
  end
