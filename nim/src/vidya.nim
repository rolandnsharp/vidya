## vidya.nim — Entry point
##
## A 103M parameter transformer with memory, written in Nim.
## Arraymancer for tensors. No PyTorch. No Python. No OCaml FFI pain.

import model, bpe, forward, train
import std/[os, times, strformat]

import std/[os]

let baseDir = getAppDir().parentDir()  # nim/ directory
let vidyaRoot = baseDir.parentDir()    # vidya/ directory
let dataFile = vidyaRoot / "chat_input.txt"
let tokenizerFile = vidyaRoot / "tokenizer_nim.bin"

when isMainModule:
  echo "vidya (nim) starting..."

  # Load or train tokenizer
  var tok: Tokenizer
  if fileExists(tokenizerFile):
    echo &"loading tokenizer from {tokenizerFile}"
    tok = loadTokenizer(tokenizerFile)
  else:
    echo &"training tokenizer on {dataFile}..."
    let docs = loadDocs(dataFile)
    echo &"loaded {docs.len} docs"
    tok = trainBpe(docs)
    saveTokenizer(tok, tokenizerFile)
  echo &"vocab size: {tok.vocab.len}"

  # Init model
  var m = initModel(tok.vocab.len)
  echo &"num params: {m.paramCount()}"

  # Load and tokenize training data
  echo "loading training data..."
  let docs = loadDocs(dataFile)
  echo &"num docs: {docs.len}"

  echo "tokenizing..."
  let tStart = cpuTime()
  var tokenizedDocs: seq[seq[int]]
  for doc in docs:
    tokenizedDocs.add(tok.encode(doc))
  echo &"tokenized {tokenizedDocs.len} docs in {cpuTime() - tStart:.2f}s"

  # Train (forward-only for now — validates architecture)
  train(m, tokenizedDocs, numSteps = 200000)
