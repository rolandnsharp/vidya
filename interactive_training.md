# Interactive Training Mode for Vidya

## Overview

The chat interface IS the training interface. The human asks a question, sees
5 responses, picks the best or types a better one, and the model trains on
it — one gradient step per interaction. No batch RL, no reward function, no
advantage estimation. The human is the reward.

A conversation context file (`context.md`) accumulates the full history of
prompts and chosen responses. The model reads this on startup so it has
multi-turn context across sessions. It isn't constantly replying from only
the last query — it sees the whole conversation so far.

## Two Interfaces

### Interactive (`--train`)

For humans sitting at the terminal:

```
--- interactive training mode (type 'quit' to exit) ---
(loaded context: 12 turns)
> hello
1: Yes . May I help you ?
2: Hello there .
3: Hi how are you ?
4: Good morning .
5: Hey .
[1-5 or type response] > 3
trained. (loss 3.42, step 1)
> what is your name?
1: It is Mr . Classic .
2: My name is not important .
3: I am a computer .
4: Yes .
5: I do not have a name .
[1-5 or type response] > My name is Mr . Classic . Nice to meet you .
trained. (loss 2.81, step 2)
> quit
saved checkpoint to ../../microgpt_chat_10m_v4_train.bin (10011648 params)
```

Load once, train interactively. Every turn appends to `context.md`.

### CLI (`--prompt` / `--teach`)

For AI tools like Claude Code to drive training programmatically:

```bash
# Step 1: prompt the model, get 5 responses
dune exec bin/main.exe -- --load --prompt "hello"
# 1: Yes . May I help you ?
# 2: Hello there .
# 3: Hi how are you ?
# 4: Good morning .
# 5: Hey .

# Step 2: teach it which response was best
dune exec bin/main.exe -- --load --teach 3

# OR teach it a better response
dune exec bin/main.exe -- --load --teach "Hi there , nice to meet you ."
```

Two steps, two calls. `--prompt` generates and saves state. `--teach` loads
state, trains, saves checkpoint. Both append to `context.md` so subsequent
calls see the full conversation history.

## The Context File

`context.md` is a plain text file that accumulates the conversation:

```
<|user|> hello <|assistant|> Hi how are you ?
<|user|> what is your name? <|assistant|> My name is Mr . Classic .
<|user|> what is your favorite color? <|assistant|> I like blue .
```

This is the same format that `Bpe.encode` already parses. Readable by
humans, parseable by the model. On startup, the context is loaded and
passed as conversation history — the model generates conditioned on everything
that came before.

The context file is what gives both modes memory across sessions:
- Interactive `--train` picks up where the last session left off
- CLI `--prompt` calls see the full conversation so far
- The model builds on previous turns, not just the latest query

## The Training Step

Both interfaces run the same function:

```
1. Encode prompt + chosen response as token sequence
2. compute_loss (standard NLL — same as SFT)
3. Tensor.backward
4. clip_grad_norm (max norm 1.0)
5. adam_step_fixed (lr = 1e-5)
```

One step. Instant on CPU. Next question.

## Why This Works

**Selection** (typing a number): reinforces the model's own best response.
Same as Expert Iteration. The model already generated it — training makes
it more likely next time.

**Typing** (providing a response): injects new knowledge. The model learns
patterns it couldn't generate on its own. Breaks the capability ceiling.

Both cases call `compute_loss` on the token sequence. The model doesn't know
whether the response came from its own generation or from a human keyboard.

## Scaling

One gradient step on one sequence is trivial. The model does 300,000 during
SFT. One more takes a fraction of a second on CPU.

**With people**: ten friends chatting via a web interface = ten gradient steps
per minute. Each person thinks they're just having a conversation.

**With bigger models**: interactive RL runs on CPU because humans are slow.
A 1B param model doing one step every 30 seconds is nothing. Train the big
model once with SFT on a rented GPU. Bring it home. Interactive training
runs on anything.

## Fast Startup

The BPE tokenizer is saved to disk after first training. Subsequent calls
load it directly instead of rebuilding from 123K docs. Startup drops from
~5 seconds to under 1 second. This makes the CLI mode responsive.

## Usage

```bash
cd ocaml/vidya

# Interactive mode (human at terminal)
dune exec bin/main.exe -- --load --train

# CLI mode (AI tool driving training)
dune exec bin/main.exe -- --load --prompt "hello"
dune exec bin/main.exe -- --load --teach 3
dune exec bin/main.exe -- --load --prompt "what is your name?"
dune exec bin/main.exe -- --load --teach "My name is Mr . Classic ."

# Check conversation history
cat context.md
```
