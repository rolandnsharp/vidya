\ microgpt.fs — Minimal GPT in Forth (after Karpathy)
\ Tape-based scalar autograd + 1-layer transformer, character-level.
\ Usage: gforth -m 64M microgpt.fs

\ ============================================================
\ CONFIGURATION
\ ============================================================
16 constant N_EMBD
32 constant BLOCK_SIZE
4  constant N_HEAD
1  constant N_LAYER
N_EMBD N_HEAD / constant HEAD_DIM
1000 constant NUM_STEPS
256 constant MAX_VOCAB
128 constant MAX_TOK
20000 constant MAX_DOCS
500000 constant MAX_TAPE

\ ============================================================
\ PRNG (xorshift32 — simple, avoids 64-bit sign issues)
\ ============================================================
variable rng-state   12345 rng-state !

: rng ( -- u )
  rng-state @
  dup 13 lshift xor
  dup 17 rshift xor
  dup 5  lshift xor
  $7FFFFFFF and
  dup rng-state ! ;

: rng-float ( F: -- f )
  rng s>f 2147483647e f/ ;

: rng-gauss ( F: -- f )
  begin rng-float fdup 1e-10 f< 0= until
  fln -2e f* fsqrt
  rng-float 6.283185307179586e f* fcos f* ;

\ ============================================================
\ HEAP HELPER
\ ============================================================
: hall ( n -- addr ) allocate throw ;

\ ============================================================
\ TAPE-BASED AUTOGRAD
\ ============================================================
0 constant OP_LEAF
1 constant OP_ADD
2 constant OP_MUL
3 constant OP_POW
4 constant OP_LOG
5 constant OP_EXP
6 constant OP_RELU

variable tape-op    MAX_TAPE cells  hall tape-op !
variable tape-arg0  MAX_TAPE cells  hall tape-arg0 !
variable tape-arg1  MAX_TAPE cells  hall tape-arg1 !
variable tape-data  MAX_TAPE floats hall tape-data !
variable tape-grad  MAX_TAPE floats hall tape-grad !

variable tape-len   0 tape-len !
variable n-params   0 n-params !

: t-op   ( i -- addr ) cells  tape-op   @ + ;
: t-arg0 ( i -- addr ) cells  tape-arg0 @ + ;
: t-arg1 ( i -- addr ) cells  tape-arg1 @ + ;
: t-data ( i -- addr ) floats tape-data @ + ;
: t-grad ( i -- addr ) floats tape-grad @ + ;

: f>bits ( F: f -- ) ( -- u ) pad f! pad @ ;
: bits>f ( u -- ) ( F: -- f ) pad ! pad f@ ;

\ --- leaf ---
: v-leaf ( F: data -- ) ( -- idx )
  tape-len @  >r
  OP_LEAF r@ t-op !
  r@ t-data f!
  0e r@ t-grad f!
  1 tape-len +!
  r> ;

: v-const ( F: v -- idx ) v-leaf ;

\ --- binary ops ---
: val+ ( a b -- c )
  over t-data f@ dup t-data f@ f+
  tape-len @ >r
  swap r@ t-arg0 ! r@ t-arg1 !
  OP_ADD r@ t-op ! r@ t-data f! 0e r@ t-grad f!
  1 tape-len +! r> ;

: val* ( a b -- c )
  over t-data f@ dup t-data f@ f*
  tape-len @ >r
  swap r@ t-arg0 ! r@ t-arg1 !
  OP_MUL r@ t-op ! r@ t-data f! 0e r@ t-grad f!
  1 tape-len +! r> ;

\ --- unary ops ---
fvariable pow-exp
: val** ( a -- c ) ( F: exp -- )
  pow-exp f!
  dup t-data f@ pow-exp f@ f**
  tape-len @ >r
  r@ t-arg0 !
  OP_POW r@ t-op ! r@ t-data f!
  pow-exp f@ f>bits r@ t-arg1 !
  0e r@ t-grad f!
  1 tape-len +! r> ;

: v-log ( a -- c )
  dup t-data f@ fln
  tape-len @ >r
  r@ t-arg0 ! OP_LOG r@ t-op ! r@ t-data f! 0e r@ t-grad f!
  1 tape-len +! r> ;

: v-exp ( a -- c )
  dup t-data f@ fexp
  tape-len @ >r
  r@ t-arg0 ! OP_EXP r@ t-op ! r@ t-data f! 0e r@ t-grad f!
  1 tape-len +! r> ;

: v-relu ( a -- c )
  dup t-data f@ fdup 0e f< if fdrop 0e then
  tape-len @ >r
  r@ t-arg0 ! OP_RELU r@ t-op ! r@ t-data f! 0e r@ t-grad f!
  1 tape-len +! r> ;

\ --- compound ---
: v-sub ( a b -- c ) -1e v-const val* val+;
: v-div ( a b -- c ) -1e val** val* ;

\ ============================================================
\ BACKWARD PASS
\ ============================================================
: bw-add ( i -- )
  >r r@ t-grad f@ fdup
  r@ t-arg0 @ t-grad dup f@ f+ f!
  r> t-arg1 @ t-grad dup f@ f+ f! ;

: bw-mul ( i -- )
  >r
  r@ t-grad f@ r@ t-arg1 @ t-data f@ f*
  r@ t-arg0 @ t-grad dup f@ f+ f!
  r@ t-grad f@ r@ t-arg0 @ t-data f@ f*
  r> t-arg1 @ t-grad dup f@ f+ f! ;

: bw-pow ( i -- )
  >r
  r@ t-arg1 @ bits>f fdup
  r@ t-arg0 @ t-data f@ fswap 1e f- f** f*
  r@ t-grad f@ f*
  r> t-arg0 @ t-grad dup f@ f+ f! ;

: bw-log ( i -- )
  >r
  r@ t-grad f@ r> t-arg0 @
  dup t-data f@ f/
  t-grad dup f@ f+ f! ;

: bw-exp ( i -- )
  >r
  r@ t-data f@ r@ t-grad f@ f*
  r> t-arg0 @ t-grad dup f@ f+ f! ;

: bw-relu ( i -- )
  >r
  r@ t-arg0 @ t-data f@ 0e fswap f<
  if r@ t-grad f@ r@ t-arg0 @ t-grad dup f@ f+ f! then
  r> drop ;

: backward ( loss-idx -- )
  1e over t-grad f!
  begin dup 0 >= while
    dup t-op @
    case
      OP_ADD  of dup bw-add  endof
      OP_MUL  of dup bw-mul  endof
      OP_POW  of dup bw-pow  endof
      OP_LOG  of dup bw-log  endof
      OP_EXP  of dup bw-exp  endof
      OP_RELU of dup bw-relu endof
    endcase
    1-
  repeat drop ;

: tape-reset-step
  tape-len @ 0 do 0e i t-grad f! loop
  n-params @ tape-len ! ;

\ ============================================================
\ FILE I/O
\ ============================================================
1100000 constant MAX_TEXT
variable text-buf  MAX_TEXT hall text-buf !
variable text-len  0 text-len !
variable doc-start MAX_DOCS cells hall doc-start !
variable doc-len   MAX_DOCS cells hall doc-len !
variable n-docs    0 n-docs !

: load-input ( -- )
  s" input.txt" r/o open-file throw { fd }
  0 text-len ! 0 n-docs !
  begin
    text-buf @ text-len @ + 256 fd read-line throw
  while
    dup 0> if
      text-len @ n-docs @ cells doc-start @ + !
      dup       n-docs @ cells doc-len   @ + !
      1 n-docs +!
    then
    text-len +!
  repeat drop
  fd close-file throw
  ." docs: " n-docs @ . cr ;

\ ============================================================
\ TOKENIZER
\ ============================================================
create char-to-id MAX_VOCAB cells allot
create id-to-char MAX_VOCAB allot
variable vocab-size  variable bos-token
create tok-buf MAX_TOK cells allot

: build-vocab ( -- )
  char-to-id MAX_VOCAB cells -1 fill
  0 { id }
  text-len @ 0 do
    text-buf @ i + c@ { ch }
    ch cells char-to-id + @ -1 = if
      ch id-to-char id + c!
      id ch cells char-to-id + !
      id 1+ to id
    then
  loop
  id bos-token !
  id 1+ vocab-size !
  ." vocab: " vocab-size @ . cr ;

: tokenize-doc ( doc-idx -- ntok )
  dup  cells doc-start @ + @ { start }
       cells doc-len   @ + @ { len }
  bos-token @ tok-buf !
  1 { pos }
  len 0 do
    text-buf @ start + i + c@ cells char-to-id + @
    tok-buf pos cells + !
    pos 1+ to pos
    pos MAX_TOK 2 - >= if leave then
  loop
  bos-token @ tok-buf pos cells + !
  pos 1+ ;

\ ============================================================
\ SHUFFLE
\ ============================================================
variable doc-order  MAX_DOCS cells hall doc-order !

: init-doc-order  n-docs @ 0 do i doc-order @ i cells + ! loop ;

: shuffle-docs
  n-docs @ 1 do
    rng i 1+ mod { j }
    doc-order @ i cells + @ { vi }
    doc-order @ j cells + @ { vj }
    vj doc-order @ i cells + !
    vi doc-order @ j cells + !
  loop ;

\ ============================================================
\ MODEL PARAMETERS
\ ============================================================
create wte-idx    MAX_VOCAB N_EMBD * cells allot
create wpe-idx    BLOCK_SIZE N_EMBD * cells allot
create lmhead-idx MAX_VOCAB N_EMBD * cells allot
create wq-idx     N_EMBD N_EMBD * cells allot
create wk-idx     N_EMBD N_EMBD * cells allot
create wv-idx     N_EMBD N_EMBD * cells allot
create wo-idx     N_EMBD N_EMBD * cells allot
create fc1-idx    N_EMBD 4 * N_EMBD * cells allot
create fc2-idx    N_EMBD N_EMBD 4 * * cells allot

\ mat@/mat!: row-major access.  index = row * ncols + col
: mat@ ( row col ncols base -- val )
  >r >r swap r> * + cells r> + @ ;
: mat! ( val row col ncols base -- )
  >r >r swap r> * + cells r> + ! ;

: init-matrix ( nrows ncols base -- )
  { base } { nc } { nr }
  nr 0 do
    nc 0 do
      0.08e rng-gauss f* v-leaf
      j i nc base mat!
    loop
  loop ;

: init-params ( -- )
  vocab-size @ N_EMBD wte-idx    init-matrix
  BLOCK_SIZE   N_EMBD wpe-idx    init-matrix
  vocab-size @ N_EMBD lmhead-idx init-matrix
  N_EMBD N_EMBD wq-idx init-matrix
  N_EMBD N_EMBD wk-idx init-matrix
  N_EMBD N_EMBD wv-idx init-matrix
  N_EMBD N_EMBD wo-idx init-matrix
  N_EMBD 4 * N_EMBD fc1-idx init-matrix
  N_EMBD N_EMBD 4 * fc2-idx init-matrix
  tape-len @ n-params !
  ." params: " n-params @ . cr ;

\ ============================================================
\ VECTOR BUFFERS & KV CACHE
\ ============================================================
N_EMBD 4 * constant MAX_VEC
create vec-x    MAX_VEC cells allot
create vec-y    MAX_VEC cells allot
create vec-z    N_EMBD  cells allot
create vec-q    N_EMBD  cells allot
create vec-k    N_EMBD  cells allot
create vec-v    N_EMBD  cells allot
create vec-attn N_EMBD  cells allot

variable kv-keys  N_LAYER BLOCK_SIZE * N_EMBD * cells hall kv-keys !
variable kv-vals  N_LAYER BLOCK_SIZE * N_EMBD * cells hall kv-vals !

\ kv-addr: compute address into KV cache
\ layout: [layer][pos][dim], strides: BLOCK_SIZE*N_EMBD, N_EMBD, 1
: kv-addr ( base-var layer pos dim -- addr )
  >r >r BLOCK_SIZE N_EMBD * * r> N_EMBD * + r> + cells
  swap @ + ;

: kv-key@ ( layer pos dim -- val ) kv-keys kv-addr @ ;
: kv-key! ( val layer pos dim -- ) kv-keys kv-addr ! ;
: kv-val@ ( layer pos dim -- val ) kv-vals kv-addr @ ;
: kv-val! ( val layer pos dim -- ) kv-vals kv-addr ! ;

\ ============================================================
\ LINEAR, RMSNORM, SOFTMAX
\ ============================================================

\ linear: y[row] = dot(W[row], x)  — reads vec-x, writes vec-y
\ Uses explicit loop vars to avoid nested do-loop index conflicts.
variable lin-row
: linear ( nout nin wbase -- )
  { w } { nin } { nout }
  nout 0 do
    i lin-row !
    \ first element of dot product
    lin-row @ 0 nin w mat@  vec-x @ val*
    nin 1 do
      lin-row @ i nin w mat@  vec-x i cells + @ val*  v+
    loop
    vec-y lin-row @ cells + !
  loop ;

\ rmsnorm: normalizes vec-x[0..n-1] in place
: rmsnorm-vec ( n -- )
  { n }
  vec-x @ dup val*
  n 1 do vec-x i cells + @ dup val* val+loop
  n s>f 1e fswap f/ v-const val*
  1e-5 v-const v+
  -0.5e val** { scale }
  n 0 do
    vec-x i cells + @ scale val*
    vec-x i cells + !
  loop ;

\ softmax over n tape-indices at src, in-place
create sm-buf MAX_VOCAB cells allot

: do-softmax ( n src -- )
  { src } { n }
  \ find max
  src @ t-data f@
  n 1 do
    src i cells + @ t-data f@
    fover fover f< if fswap then fdrop
  loop  \ FS: maxval
  \ exp(x_i - max)
  n 0 do
    src i cells + @
    fdup v-const v-sub v-exp
    sm-buf i cells + !
  loop fdrop
  \ sum
  sm-buf @
  n 1 do sm-buf i cells + @ val+loop  { total }
  \ divide
  n 0 do
    sm-buf i cells + @ total v-div
    src i cells + !
  loop ;

\ ============================================================
\ ATTENTION  (explicit counter variables to avoid nesting issues)
\ ============================================================
variable cur-layer
variable cur-head
variable cur-t
variable cur-d
create attn-logits BLOCK_SIZE cells allot

: attn-dot ( pos hs -- )
  \ compute attn logits for positions 0..pos, head starting at hs
  { hs } { pos }
  pos 1+ 0 do
    i cur-t !
    vec-q hs cells + @
    cur-layer @ cur-t @ hs kv-key@  val*
    HEAD_DIM 1 do
      vec-q hs i + cells + @
      cur-layer @ cur-t @ hs i + kv-key@  val* v+
    loop
    HEAD_DIM s>f fsqrt 1e fswap f/ v-const val*
    attn-logits cur-t @ cells + !
  loop ;

: attn-mix ( pos hs -- )
  \ weighted sum of values for this head
  { hs } { pos }
  HEAD_DIM 0 do
    i cur-d !
    attn-logits @
    cur-layer @ 0 hs cur-d @ + kv-val@  val*
    pos 1+ 1 do
      attn-logits i cells + @
      cur-layer @ i hs cur-d @ + kv-val@  val* v+
    loop
    vec-attn hs cur-d @ + cells + !
  loop ;

: do-attention ( pos -- )
  { pos }
  N_HEAD 0 do
    i cur-head !
    cur-head @ HEAD_DIM * { hs }
    pos hs attn-dot
    pos 1+ attn-logits do-softmax
    pos hs attn-mix
  loop ;

\ ============================================================
\ GPT FORWARD (single token) — result logits in vec-y
\ ============================================================
: gpt-forward ( tok pos -- )
  { pos } { tok }
  \ token + position embedding
  N_EMBD 0 do
    tok i N_EMBD wte-idx mat@
    pos i N_EMBD wpe-idx mat@ v+
    vec-x i cells + !
  loop
  N_EMBD rmsnorm-vec

  N_LAYER 0 do
    i cur-layer !
    \ save residual
    N_EMBD 0 do vec-x j cells + @ vec-z j cells + ! loop
    N_EMBD rmsnorm-vec
    \ Q K V
    N_EMBD N_EMBD wq-idx linear
    N_EMBD 0 do vec-y j cells + @ vec-q j cells + ! loop
    N_EMBD N_EMBD wk-idx linear
    N_EMBD 0 do vec-y j cells + @ vec-k j cells + ! loop
    N_EMBD N_EMBD wv-idx linear
    N_EMBD 0 do vec-y j cells + @ vec-v j cells + ! loop
    \ store KV cache
    N_EMBD 0 do
      vec-k j cells + @ cur-layer @ pos j kv-key!
      vec-v j cells + @ cur-layer @ pos j kv-val!
    loop
    \ attention
    pos do-attention
    \ project + residual
    N_EMBD 0 do vec-attn j cells + @ vec-x j cells + ! loop
    N_EMBD N_EMBD wo-idx linear
    N_EMBD 0 do vec-y j cells + @ vec-z j cells + @ val+vec-x j cells + ! loop
    \ MLP: save residual
    N_EMBD 0 do vec-x j cells + @ vec-z j cells + ! loop
    N_EMBD rmsnorm-vec
    N_EMBD 4 * N_EMBD fc1-idx linear
    N_EMBD 4 * 0 do vec-y j cells + @ v-relu vec-x j cells + ! loop
    N_EMBD N_EMBD 4 * fc2-idx linear
    N_EMBD 0 do vec-y j cells + @ vec-z j cells + @ val+vec-x j cells + ! loop
  loop
  \ final projection
  vocab-size @ N_EMBD lmhead-idx linear ;

\ ============================================================
\ ADAM OPTIMIZER
\ ============================================================
variable adam-m  variable adam-v

: init-adam
  MAX_TAPE floats hall adam-m !
  MAX_TAPE floats hall adam-v !
  MAX_TAPE 0 do
    0e adam-m @ i floats + f!
    0e adam-v @ i floats + f!
  loop ;

fvariable lr-t

: adam-update ( step -- )
  { step }
  1e step s>f NUM_STEPS s>f f/ f- 0.01e f* lr-t f!
  n-params @ 0 do
    i t-grad f@ { f: g }
    \ m = beta1*m + (1-beta1)*g
    0.85e adam-m @ i floats + f@ f*  0.15e g f* f+
    fdup adam-m @ i floats + f!   { f: mnew }
    \ v = beta2*v + (1-beta2)*g^2
    0.99e adam-v @ i floats + f@ f*  0.01e g g f* f* f+
    fdup adam-v @ i floats + f!   { f: vnew }
    \ bias-corrected
    mnew 1e 0.85e step 1+ s>f f** f- f/  { f: mhat }
    vnew 1e 0.99e step 1+ s>f f** f- f/  { f: vhat }
    \ update: data -= lr * mhat / (sqrt(vhat) + eps)
    i t-data f@
    lr-t f@ mhat f* vhat fsqrt 1e-8 f+ f/
    f-
    i t-data f!
  loop ;

\ ============================================================
\ TRAINING
\ ============================================================
variable step-var

: train ( -- )
  load-input
  build-vocab
  init-params
  init-adam
  init-doc-order
  shuffle-docs
  ." Training " NUM_STEPS . ." steps..." cr
  NUM_STEPS 0 do
    i step-var !
    tape-reset-step
    \ pick doc
    doc-order @ step-var @ n-docs @ mod cells + @ { doc }
    doc tokenize-doc { ntok }
    ntok 1- BLOCK_SIZE min { n }
    n 0> if
      0 { loss-idx }
      n 0 do
        tok-buf j cells + @ j gpt-forward
        vocab-size @ vec-y do-softmax
        \ -log(prob[target])
        tok-buf j 1+ cells + @ cells vec-y + @
        v-log -1e v-const val*
        j 0= if else loss-idx val+then
        to loss-idx
      loop
      \ average
      n s>f 1e fswap f/ v-const loss-idx val* to loss-idx
      loss-idx backward
      step-var @ adam-update
      step-var @ 50 mod 0= if
        ." step " step-var @ 1+ .
        ." | loss " loss-idx t-data f@ f. cr
      then
    then
  loop
  ." Done." cr ;

\ ============================================================
\ INFERENCE (raw floats, no autograd)
\ ============================================================
create inf-buf MAX_VOCAB floats allot

: inference-softmax ( -- )
  vec-y @ t-data f@ { f: mx }
  vocab-size @ 1 do
    vec-y i cells + @ t-data f@
    fdup mx f> if to mx else fdrop then
  loop
  0e { f: total }
  vocab-size @ 0 do
    vec-y i cells + @ t-data f@ mx f- 0.5e f/ fexp
    fdup inf-buf i floats + f!
    total f+ to total
  loop
  vocab-size @ 0 do
    inf-buf i floats + f@ total f/ inf-buf i floats + f!
  loop ;

: sample-from-probs ( -- token-id )
  rng-float { f: r }
  0e { f: cum }
  vocab-size @ 0 do
    inf-buf i floats + f@ cum f+ to cum
    cum r f>= if i unloop exit then
  loop
  vocab-size @ 1- ;

: generate ( -- )
  ." --- hallucinated Plotinus ---" cr
  20 0 do
    tape-reset-step
    bos-token @ { tok }
    ." > "
    BLOCK_SIZE 0 do
      tok j gpt-forward
      inference-softmax
      sample-from-probs to tok
      tok bos-token @ = if leave then
      tok id-to-char + c@ emit
    loop
    cr
  loop ;

\ ============================================================
\ MAIN
\ ============================================================
: main train generate ;
main bye
