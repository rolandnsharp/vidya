(* forth.ml — Minimal Forth interpreter as symbolic AI substrate
   ==============================================================

   A real Forth interpreter with dictionary, stack, and stack-effect
   validation. The dictionary serves as structured external memory:
   concepts extracted from the corpus become Forth words with associations,
   and these associations bias generation toward coherent text.

   Not a full Forth — just enough to be a useful symbolic substrate.
   No control flow (IF/THEN/ELSE, loops) yet — those come in Stage 15
   when the model starts proposing its own word definitions.

   Three kinds of dictionary entries:
   - Primitive: built-in stack operations (DUP, +, etc.)
   - Defined: user/model-defined words (body is a list of word names)
   - Concept: corpus-derived concepts with associations and token mappings

   Key property from FORTH_AS_SYMBOLIC_AI.md:
     "Forth verification is O(n) in definition length, always terminates,
      and catches the most common error classes." *)

(* ── Types ──────────────────────────────────────────────────────── *)

(* Stack effect: how many values a word consumes and produces.
   DUP is { consumed=1; produced=2 }.
   + is { consumed=2; produced=1 }.
   DROP is { consumed=1; produced=0 }. *)
type stack_effect = { consumed : int; produced : int }

(* Dictionary entry kinds *)
type entry_kind =
  | Primitive of (float list -> float list)
  | Defined of string list
  | Concept of {
      associations : (string * float) list;   (* (name, weight) *)
      strength : float;
      token_ids : int list;
    }

type entry = {
  name : string;
  kind : entry_kind;
  effect : stack_effect;
}

type dictionary = {
  entries : (string, entry) Hashtbl.t;
  mutable generation : int;
}

(* Raised by validate_definition on invalid word bodies *)
exception Validation_error of string

(* ── Dictionary operations ──────────────────────────────────────── *)

let lookup dict name =
  Hashtbl.find_opt dict.entries name

let define dict name kind effect =
  Hashtbl.replace dict.entries name { name; kind; effect };
  dict.generation <- dict.generation + 1

let forget dict name =
  Hashtbl.remove dict.entries name;
  dict.generation <- dict.generation + 1

(* ── Stack-effect validation ────────────────────────────────────── *)

(* Left-to-right scan of a word body. For each word:
   1. Look up its stack effect in the dictionary
   2. If depth < consumed, we need more inputs (track max_consumed)
   3. Update running depth: depth - consumed + produced

   Returns the net stack effect of the entire body.
   Raises Validation_error on unknown words. *)
let validate_definition dict body =
  let max_consumed = ref 0 in
  let depth = ref 0 in
  List.iter (fun word ->
    match lookup dict word with
    | None ->
      raise (Validation_error (Printf.sprintf "unknown word: %s" word))
    | Some entry ->
      if !depth < entry.effect.consumed then
        max_consumed := !max_consumed + (entry.effect.consumed - !depth);
      depth := max 0 (!depth - entry.effect.consumed) + entry.effect.produced
  ) body;
  Ok { consumed = !max_consumed; produced = !depth }

(* Wrapper that catches the exception for callers who prefer result *)
let validate dict body =
  try validate_definition dict body
  with Validation_error msg -> Error msg

(* ── Concept queries ────────────────────────────────────────────── *)

(* Direct associations of a concept word — returns (name, weight) pairs *)
let associations dict name =
  match lookup dict name with
  | Some { kind = Concept { associations; _ }; _ } -> associations
  | _ -> []

(* Just the names of direct associations (for BFS traversal) *)
let association_names dict name =
  List.map fst (associations dict name)

(* BFS to depth N, collecting all reachable concept names.
   Avoids cycles by tracking visited set. *)
let deep_associations dict name max_depth =
  let visited = Hashtbl.create 32 in
  let result = ref [] in
  let queue = Queue.create () in
  Queue.add (name, 0) queue;
  Hashtbl.replace visited name ();
  while not (Queue.is_empty queue) do
    let (current, depth) = Queue.pop queue in
    if depth > 0 then  (* don't include the starting word *)
      result := current :: !result;
    if depth < max_depth then begin
      let assocs = association_names dict current in
      List.iter (fun assoc ->
        if not (Hashtbl.mem visited assoc) then begin
          Hashtbl.replace visited assoc ();
          Queue.add (assoc, depth + 1) queue
        end
      ) assocs
    end
  done;
  !result

(* BPE token IDs that correspond to a concept *)
let concept_token_ids dict name =
  match lookup dict name with
  | Some { kind = Concept { token_ids; _ }; _ } -> token_ids
  | _ -> []

(* Strength (prominence) of a concept *)
let concept_strength dict name =
  match lookup dict name with
  | Some { kind = Concept { strength; _ }; _ } -> strength
  | _ -> 0.0

(* All concept entries in the dictionary *)
let all_concepts dict =
  Hashtbl.fold (fun _name entry acc ->
    match entry.kind with
    | Concept _ -> entry :: acc
    | _ -> acc
  ) dict.entries []

(* Update a single association weight for a concept *)
let update_association_weight dict concept_name assoc_name new_weight =
  match lookup dict concept_name with
  | Some ({ kind = Concept ({ associations; _ } as c); _ } as entry) ->
    let updated = List.map (fun (n, w) ->
      if n = assoc_name then (n, new_weight) else (n, w)
    ) associations in
    Hashtbl.replace dict.entries concept_name
      { entry with kind = Concept { c with associations = updated } }
  | _ -> ()

(* Batch update association weights for a concept *)
let update_association_weights dict concept_name updates =
  match lookup dict concept_name with
  | Some ({ kind = Concept ({ associations; _ } as c); _ } as entry) ->
    let updated = List.map (fun (n, w) ->
      match List.assoc_opt n updates with
      | Some new_w -> (n, new_w)
      | None -> (n, w)
    ) associations in
    Hashtbl.replace dict.entries concept_name
      { entry with kind = Concept { c with associations = updated } }
  | _ -> ()

(* ── Primitives ─────────────────────────────────────────────────── *)

(* Register a primitive word in the dictionary *)
let add_primitive dict name consumed produced fn =
  define dict name (Primitive fn) { consumed; produced }

(* Create a fresh dictionary with all primitive words.
   These are the building blocks for all definitions. *)
let create () =
  let dict = { entries = Hashtbl.create 64; generation = 0 } in

  (* Stack manipulation *)
  add_primitive dict "DUP" 1 2 (fun stack ->
    match stack with x :: rest -> x :: x :: rest | _ -> stack);
  add_primitive dict "DROP" 1 0 (fun stack ->
    match stack with _ :: rest -> rest | _ -> stack);
  add_primitive dict "SWAP" 2 2 (fun stack ->
    match stack with a :: b :: rest -> b :: a :: rest | _ -> stack);
  add_primitive dict "OVER" 2 3 (fun stack ->
    match stack with a :: b :: rest -> b :: a :: b :: rest | _ -> stack);
  add_primitive dict "ROT" 3 3 (fun stack ->
    match stack with a :: b :: c :: rest -> c :: a :: b :: rest | _ -> stack);
  add_primitive dict "NIP" 2 1 (fun stack ->
    match stack with a :: _ :: rest -> a :: rest | _ -> stack);

  (* Arithmetic *)
  add_primitive dict "+" 2 1 (fun stack ->
    match stack with a :: b :: rest -> (b +. a) :: rest | _ -> stack);
  add_primitive dict "-" 2 1 (fun stack ->
    match stack with a :: b :: rest -> (b -. a) :: rest | _ -> stack);
  add_primitive dict "*" 2 1 (fun stack ->
    match stack with a :: b :: rest -> (b *. a) :: rest | _ -> stack);
  add_primitive dict "/" 2 1 (fun stack ->
    match stack with a :: b :: rest ->
      if a <> 0.0 then (b /. a) :: rest else stack
    | _ -> stack);
  add_primitive dict "NEGATE" 1 1 (fun stack ->
    match stack with a :: rest -> (-. a) :: rest | _ -> stack);
  add_primitive dict "ABS" 1 1 (fun stack ->
    match stack with a :: rest -> (abs_float a) :: rest | _ -> stack);
  add_primitive dict "MIN" 2 1 (fun stack ->
    match stack with a :: b :: rest -> (min a b) :: rest | _ -> stack);
  add_primitive dict "MAX" 2 1 (fun stack ->
    match stack with a :: b :: rest -> (max a b) :: rest | _ -> stack);

  (* Comparison (Forth convention: -1 for true, 0 for false) *)
  add_primitive dict "=" 2 1 (fun stack ->
    match stack with a :: b :: rest ->
      (if b = a then -1.0 else 0.0) :: rest | _ -> stack);
  add_primitive dict "<" 2 1 (fun stack ->
    match stack with a :: b :: rest ->
      (if b < a then -1.0 else 0.0) :: rest | _ -> stack);
  add_primitive dict ">" 2 1 (fun stack ->
    match stack with a :: b :: rest ->
      (if b > a then -1.0 else 0.0) :: rest | _ -> stack);

  (* Logic *)
  add_primitive dict "AND" 2 1 (fun stack ->
    match stack with a :: b :: rest ->
      (if a <> 0.0 && b <> 0.0 then -1.0 else 0.0) :: rest | _ -> stack);
  add_primitive dict "OR" 2 1 (fun stack ->
    match stack with a :: b :: rest ->
      (if a <> 0.0 || b <> 0.0 then -1.0 else 0.0) :: rest | _ -> stack);
  add_primitive dict "NOT" 1 1 (fun stack ->
    match stack with a :: rest ->
      (if a = 0.0 then -1.0 else 0.0) :: rest | _ -> stack);

  dict

(* ── Execution ──────────────────────────────────────────────────── *)

(* Execute a list of words against a float stack.
   For primitives, apply the function directly.
   For defined words, recursively execute the body.
   For concepts, push the strength value onto the stack. *)
let rec execute dict words stack =
  List.fold_left (fun stk word ->
    match lookup dict word with
    | None -> stk  (* unknown word — skip *)
    | Some { kind = Primitive fn; _ } -> fn stk
    | Some { kind = Defined body; _ } -> execute dict body stk
    | Some { kind = Concept { strength; _ }; _ } -> strength :: stk
  ) stack words

(* ── Dictionary introspection ───────────────────────────────────── *)

(* List all word names in the dictionary (like Forth's WORDS) *)
let words dict =
  Hashtbl.fold (fun name _ acc -> name :: acc) dict.entries []
  |> List.sort String.compare

(* Count of entries by kind *)
let stats dict =
  let n_prim = ref 0 in
  let n_def = ref 0 in
  let n_concept = ref 0 in
  let n_assoc = ref 0 in
  Hashtbl.iter (fun _name entry ->
    match entry.kind with
    | Primitive _ -> incr n_prim
    | Defined _ -> incr n_def
    | Concept { associations; _ } ->
      incr n_concept;
      n_assoc := !n_assoc + List.length associations
  ) dict.entries;
  (!n_prim, !n_def, !n_concept, !n_assoc)
