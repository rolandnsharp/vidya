(* evolve.ml — Evolution loop: propose → validate → accumulate
   =============================================================

   The core experiment loop:
   1. Build prompt from current dictionary state + task
   2. Ask the model to propose a Forth word definition
   3. Parse the definition from the response
   4. Validate via stack effect analysis (O(n), always terminates)
   5. If valid, compile into dictionary
   6. Log everything *)

(* Parse a Forth definition from model output.
   Looks for : NAME body ; pattern. Returns (name, body_words) or None. *)
let parse_definition text =
  let text = String.trim text in
  (* Find the colon-space start *)
  match String.index_opt text ':' with
  | None -> None
  | Some colon_pos ->
    (* Find the semicolon end *)
    match String.index_opt text ';' with
    | None -> None
    | Some semi_pos when semi_pos <= colon_pos -> None
    | Some semi_pos ->
      let body_str = String.sub text (colon_pos + 1) (semi_pos - colon_pos - 1) in
      let words = String.split_on_char ' ' (String.trim body_str)
        |> List.filter (fun w -> String.length w > 0) in
      match words with
      | [] -> None
      | name :: body -> Some (String.uppercase_ascii name, body)

(* Format the dictionary for inclusion in a prompt. *)
let dictionary_prompt dict =
  let buf = Buffer.create 256 in
  Buffer.add_string buf "Available words:\n";
  let words = Forth.words dict in
  List.iter (fun name ->
    match Forth.lookup dict name with
    | None -> ()
    | Some entry ->
      Buffer.add_string buf (Printf.sprintf "  %s ( %d in -- %d out )\n"
        name entry.Forth.effect.consumed entry.effect.produced)
  ) words;
  Buffer.contents buf

(* Build a prompt for the model. *)
let build_prompt dict task =
  Printf.sprintf
{|You are a Forth programmer. Define a word using only the available words.

%s
Task: %s

Respond with ONLY the definition in this format:
: NAME body ;

Example:
: SQUARE DUP * ;
|}
    (dictionary_prompt dict) task

(* One evolution step. Returns (accepted, log_entry). *)
type log_entry = {
  iteration : int;
  task : string;
  proposal : string;
  parsed_name : string option;
  parsed_body : string list option;
  validation : (Forth.stack_effect, string) result option;
  accepted : bool;
  dict_size : int;
  elapsed_us : int;
}

let step dict _ctx iteration task =
  let t0 = Sys.time () in
  let prompt = build_prompt dict task in
  (* TODO: replace with actual model call *)
  let response = ignore prompt; failwith "TODO: llama.cpp not yet connected" in
  let elapsed_us = int_of_float ((Sys.time () -. t0) *. 1_000_000.0) in
  match parse_definition response with
  | None ->
    { iteration; task; proposal = response;
      parsed_name = None; parsed_body = None;
      validation = None; accepted = false;
      dict_size = List.length (Forth.words dict);
      elapsed_us }
  | Some (name, body) ->
    let result = Forth.validate dict body in
    let accepted = match result with Ok _ -> true | Error _ -> false in
    if accepted then begin
      match result with
      | Ok effect ->
        Forth.define dict name (Forth.Defined body) effect
      | Error _ -> ()
    end;
    { iteration; task; proposal = response;
      parsed_name = Some name;
      parsed_body = Some body;
      validation = Some result;
      accepted;
      dict_size = List.length (Forth.words dict);
      elapsed_us }
