(* bpe.ml — Byte-Pair Encoding tokenizer
   ========================================

   Learns a subword vocabulary by iteratively merging the most frequent
   adjacent pair in the corpus. The merge priority is implicit: lower
   merged_id = learned earlier = higher priority.

   Data structures packed into record type t:
     vocab     : string array        — token_id → token string (for decode)
     merges    : (int*int, int) Hashtbl — (left_id, right_id) → merged_id
     char_to_id: (string, int) Hashtbl — single-char string → initial token ID
     bos_id    : int                 — beginning-of-sequence token ID
     user_id   : int                 — <|user|> turn marker token ID
     assistant_id : int              — <|assistant|> turn marker token ID
     vocab_size: int                 — total vocabulary size *)

type t = {
  vocab : string array;
  merges : (int * int, int) Hashtbl.t;
  char_to_id : (string, int) Hashtbl.t;
  bos_id : int;
  user_id : int;
  assistant_id : int;
  vocab_size : int;
}

(* Special token markers for chat turns *)
let user_marker = "<|user|>"
let assistant_marker = "<|assistant|>"

(* Replace all occurrences of pattern with replacement in s *)
let replace_all ~pattern ~replacement s =
  let plen = String.length pattern in
  let slen = String.length s in
  if plen = 0 then s
  else begin
    let buf = Buffer.create slen in
    let i = ref 0 in
    while !i < slen do
      if !i + plen <= slen && String.sub s !i plen = pattern then begin
        Buffer.add_string buf replacement;
        i := !i + plen
      end else begin
        Buffer.add_char buf s.[!i];
        incr i
      end
    done;
    Buffer.contents buf
  end

let n_merges = 2000

(* train: Learn BPE merges from the training corpus.
   1. Build char vocab from all unique characters in docs
   2. Tokenize entire corpus as char IDs (flat array, -1 as doc separator)
   3. For each merge round: count pairs, merge most frequent

   OPTIMIZATION: Uses a flat int array for pair counts instead of Hashtbl.
   Array size: max_id² ≈ 2100² = 4,410,000 ints ≈ 35MB.
   Zeroing each round via Array.fill is a single fast memset. *)
let train docs n_merges =
  (* Strip special tokens from docs before BPE training —
     they get their own IDs, not built from characters *)
  let clean_docs = Array.map (fun doc ->
    replace_all ~pattern:user_marker ~replacement:" "
      (replace_all ~pattern:assistant_marker ~replacement:" " doc)
  ) docs in

  (* Step 1: Collect unique characters and assign initial IDs *)
  let char_set = Hashtbl.create 128 in
  Array.iter (fun doc ->
    String.iter (fun ch -> Hashtbl.replace char_set ch ()) doc
  ) clean_docs;
  let chars =
    Hashtbl.fold (fun ch () acc -> ch :: acc) char_set []
    |> List.sort Char.compare
    |> Array.of_list in
  let n_chars = Array.length chars in
  let char_to_id = Hashtbl.create n_chars in
  Array.iteri (fun i ch ->
    Hashtbl.add char_to_id (String.make 1 ch) i
  ) chars;

  (* vocab array: maps token ID → surface string.
     IDs 0..n_chars-1 are single characters.
     IDs n_chars..n_chars+n_merges-1 are merged tokens.
     ID n_chars+n_merges is BOS.
     ID n_chars+n_merges+1 is <|user|>.
     ID n_chars+n_merges+2 is <|assistant|>. *)
  let vocab = Array.make (n_chars + n_merges + 3) "" in
  Array.iteri (fun i ch -> vocab.(i) <- String.make 1 ch) chars;

  (* Step 2: Build flat corpus array with -1 separators between docs.
     The separator prevents merges across document boundaries.
     Uses clean_docs (special tokens stripped) for BPE training. *)
  let total_len = ref 0 in
  Array.iter (fun doc ->
    if !total_len > 0 then incr total_len;
    total_len := !total_len + String.length doc
  ) clean_docs;
  let corpus = Array.make !total_len (-1) in
  let pos = ref 0 in
  Array.iter (fun doc ->
    if !pos > 0 then begin corpus.(!pos) <- -1; incr pos end;
    String.iter (fun ch ->
      corpus.(!pos) <- Hashtbl.find char_to_id (String.make 1 ch);
      incr pos
    ) doc
  ) clean_docs;
  let corpus_len = ref !total_len in

  (* Step 3: Iteratively merge the most frequent pair *)
  let max_id = n_chars + n_merges + 3 in
  let pair_counts = Array.make (max_id * max_id) 0 in
  let merges = Hashtbl.create (n_merges * 2) in
  let actual_merges = ref 0 in
  (try
    for merge_round = 0 to n_merges - 1 do
      Array.fill pair_counts 0 (max_id * max_id) 0;

      for i = 0 to !corpus_len - 2 do
        let a = corpus.(i) and b = corpus.(i + 1) in
        if a >= 0 && b >= 0 then begin
          let idx = a * max_id + b in
          pair_counts.(idx) <- pair_counts.(idx) + 1
        end
      done;

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
        raise Exit
      else begin
        let a = !best_a and b = !best_b in
        let new_id = n_chars + merge_round in
        Hashtbl.replace merges (a, b) new_id;
        vocab.(new_id) <- vocab.(a) ^ vocab.(b);
        actual_merges := merge_round + 1;

        (* Replace all (a, b) with new_id in corpus — in-place *)
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
  let user_id = bos_id + 1 in
  vocab.(user_id) <- user_marker;
  let assistant_id = bos_id + 2 in
  vocab.(assistant_id) <- assistant_marker;
  let vocab_size = assistant_id + 1 in

  (* Print BPE training stats *)
  let n_docs = Array.length clean_docs in
  let n_separators = max 0 (n_docs - 1) in
  let orig_chars = !total_len - n_separators in
  let final_tokens = !corpus_len - n_separators in
  let ratio = float_of_int orig_chars /. float_of_int (max 1 final_tokens) in
  Printf.printf "BPE: %d chars + %d merges = %d vocab | %.1f chars/token\n%!"
    n_chars !actual_merges vocab_size ratio;

  { vocab; merges; char_to_id; bos_id; user_id; assistant_id; vocab_size }

(* encode_raw: Core BPE encoding of a text segment (no special tokens).
   Converts chars to IDs, applies merges, returns token ID array.
   Unknown characters are skipped gracefully. *)
let encode_raw tok text =
  let n = String.length text in
  if n = 0 then [||]
  else begin
    (* Map chars to IDs, skipping unknown chars *)
    let ids = ref [] in
    for i = n - 1 downto 0 do
      match Hashtbl.find_opt tok.char_to_id (String.make 1 text.[i]) with
      | Some id -> ids := id :: !ids
      | None -> ()
    done;
    let tokens = Array.of_list !ids in
    let len = ref (Array.length tokens) in
    if !len = 0 then [||]
    else begin
      let changed = ref true in
      while !changed do
        changed := false;
        let best_mid = ref max_int in
        let best_a = ref 0 and best_b = ref 0 in
        for i = 0 to !len - 2 do
          match Hashtbl.find_opt tok.merges (tokens.(i), tokens.(i + 1)) with
          | Some mid when mid < !best_mid ->
            best_mid := mid;
            best_a := tokens.(i);
            best_b := tokens.(i + 1)
          | _ -> ()
        done;

        if !best_mid < max_int then begin
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
      Array.sub tokens 0 !len
    end
  end

(* encode: Tokenize a string using learned BPE merges.
   Handles special tokens (<|user|>, <|assistant|>) by splitting on them,
   encoding each text segment with BPE, and inserting special token IDs.
   Result is wrapped with BOS at start and end. *)
let encode tok doc =
  let text_len = String.length doc in
  if text_len = 0 then [|tok.bos_id; tok.bos_id|]
  else begin
    (* Build result list in reverse, then reverse at the end *)
    let result = ref [tok.bos_id] in
    let add_segment text =
      let encoded = encode_raw tok text in
      let segment_list = Array.to_list encoded in
      result := List.rev_append segment_list !result
    in
    let user_len = String.length user_marker in
    let asst_len = String.length assistant_marker in
    let i = ref 0 in
    let seg_start = ref 0 in
    while !i < text_len do
      if !i + user_len <= text_len
         && String.sub doc !i user_len = user_marker then begin
        if !i > !seg_start then
          add_segment (String.sub doc !seg_start (!i - !seg_start));
        result := tok.user_id :: !result;
        i := !i + user_len;
        seg_start := !i
      end else if !i + asst_len <= text_len
                  && String.sub doc !i asst_len = assistant_marker then begin
        if !i > !seg_start then
          add_segment (String.sub doc !seg_start (!i - !seg_start));
        result := tok.assistant_id :: !result;
        i := !i + asst_len;
        seg_start := !i
      end else
        incr i
    done;
    if !seg_start < text_len then
      add_segment (String.sub doc !seg_start (text_len - !seg_start));
    result := tok.bos_id :: !result;
    List.rev !result |> Array.of_list
  end

(* decode: Convert token IDs back to text, skipping special tokens. *)
let decode tok tokens =
  let buf = Buffer.create 256 in
  Array.iter (fun id ->
    if id <> tok.bos_id && id <> tok.user_id && id <> tok.assistant_id then
      Buffer.add_string buf tok.vocab.(id)
  ) tokens;
  String.trim (Buffer.contents buf)

(* save_tokenizer: Save trained BPE tokenizer to binary file.
   Avoids retraining from corpus on every invocation. *)
let save_tokenizer filename tok =
  let oc = open_out_bin filename in
  Marshal.to_channel oc (tok : t) [Marshal.No_sharing];
  close_out oc;
  Printf.printf "saved tokenizer to %s (%d vocab)\n%!" filename tok.vocab_size

(* load_tokenizer: Load BPE tokenizer from binary file. *)
let load_tokenizer filename =
  let ic = open_in_bin filename in
  let tok : t = Marshal.from_channel ic in
  close_in ic;
  Printf.printf "loaded tokenizer from %s (%d vocab)\n%!" filename tok.vocab_size;
  tok
