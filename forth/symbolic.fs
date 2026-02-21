\ symbolic.fs — Dictionary with stack effects + stack effect validator
\ Core of the Forth symbolic AI engine

\ ============================================================
\ Dictionary Data Structure
\ ============================================================
\ Entry layout (contiguous allocated block):
\   +0:  link       (cell — next entry in hash chain)
\   +1c: name-len   (cell)
\   +2c: name chars  (cell-aligned)
\   ...: consumed   (cell)
\   ...: produced   (cell)

256 constant DICT-BUCKETS
create dict-table DICT-BUCKETS cells allot
dict-table DICT-BUCKETS cells erase

\ --- helpers ---

: cell-align ( n -- n' ) cell 1- + cell negate and ;

: dict-hash ( c-addr u -- index )
  0 -rot 0 ?do dup i + c@ rot + swap loop drop 255 and ;

\ --- entry field accessors ---

: entry>namelen ( entry -- addr ) cell+ ;
: entry>name ( entry -- c-addr u )
  entry>namelen dup @ swap cell+ swap ;
: entry>fx ( entry -- addr-of-consumed )
  dup entry>name nip cell-align + cell+ cell+ ;
: entry>consumed ( entry -- n ) entry>fx @ ;
: entry>produced ( entry -- n ) entry>fx cell+ @ ;
: dict-effect ( entry -- consumed produced )
  dup entry>consumed swap entry>produced ;

\ --- creation ---

: dict-alloc ( c-addr u consumed produced -- entry )
  { ca len con pro }
  len cell-align cell + cell + cell + cell + allocate throw
  { ent }
  len ent entry>namelen !
  ca ent entry>namelen cell+ len cmove
  con ent entry>fx !
  pro ent entry>fx cell+ !
  ent ;

\ --- lookup (exact match) ---

: dict-find ( c-addr u -- entry | 0 )
  2dup dict-hash cells dict-table + @
  begin dup while
    >r 2dup r@ entry>name compare 0= if 2drop r> exit then
    r> @
  repeat nip nip ;

\ --- case-insensitive helpers ---

: upchar ( c -- C ) dup [char] a >= over [char] z <= and if 32 - then ;

: istr= ( c1 u1 c2 u2 -- flag )
  rot over <> if 2drop drop false exit then
  0 ?do
    over i + c@ upchar over i + c@ upchar
    <> if 2drop false unloop exit then
  loop 2drop true ;

: dict-find-i ( c-addr u -- entry | 0 )
  2dup dict-find dup if nip nip exit then drop
  DICT-BUCKETS 0 do
    i cells dict-table + @
    begin dup while
      >r 2dup r@ entry>name istr= if 2drop r> unloop exit then
      r> @
    repeat drop
  loop 2drop 0 ;

\ --- insertion ---

: dict-add ( c-addr u consumed produced -- )
  2over dict-find if 2drop 2drop drop exit then
  { ca len con pro }
  ca len con pro dict-alloc { ent }
  ca len dict-hash cells dict-table + { bucket }
  bucket @ ent !
  ent bucket ! ;

\ --- listing ---

: dict-list ( -- )
  DICT-BUCKETS 0 do
    i cells dict-table + @
    begin dup while
      dup entry>name type
      ."  ( " dup entry>consumed 0 .r ."  -- " dup entry>produced 0 .r ."  )  "
      @
    repeat drop
  loop cr ;

\ ============================================================
\ Numeric Literal Detection
\ ============================================================

: is-digit? ( c -- flag ) dup [char] 0 >= swap [char] 9 <= and ;

: parse-number? ( c-addr u -- n true | false )
  dup 0= if 2drop false exit then
  over c@ [char] - = if
    1 /string dup 0= if 2drop false exit then
    0 -rot 0 ?do
      dup i + c@ dup is-digit? 0= if drop 2drop false unloop exit then
      [char] 0 - swap 10 * + swap
    loop drop negate true
  else
    0 -rot 0 ?do
      dup i + c@ dup is-digit? 0= if drop 2drop false unloop exit then
      [char] 0 - swap 10 * + swap
    loop drop true
  then ;

\ ============================================================
\ String Parsing Helpers
\ ============================================================

: skip-spaces ( c-addr u -- c-addr' u' )
  begin dup 0> while over c@ bl = while 1 /string repeat then ;

\ Parse next whitespace-delimited token.
: next-word ( c-addr u -- c-addr' u' tok tok-len )
  skip-spaces
  dup 0= if 0 0 exit then
  2dup  ( orig-addr orig-len orig-addr orig-len )
  begin dup 0> while over c@ bl <> while 1 /string repeat then
  \ Stack: ( orig-addr orig-len adv-addr adv-len )
  \ remaining = (adv-addr, adv-len)
  \ token    = (orig-addr, orig-len - adv-len)
  2swap ( adv-addr adv-len orig-addr orig-len )
  2 pick -  ( adv-addr adv-len orig-addr token-len )
;

\ ============================================================
\ Stack Effect Validator
\ ============================================================

256 constant MAX-NAME
create def-name MAX-NAME allot
variable def-name-len

\ validate: parse ": NAME body ;" and compute stack effect.
\ Returns:  consumed produced true   OR   false
: validate ( c-addr u -- consumed produced true | false )
  \ Parse colon
  next-word ( rem reml tok tokl )
  dup 0= if 2drop 2drop false exit then
  over c@ [char] : = over 1 = and 0= if 2drop 2drop false exit then
  2drop ( rem reml )

  \ Parse name
  next-word ( rem reml tok tokl )
  dup 0= if 2drop 2drop false exit then
  dup def-name-len !
  { rr rl namep namel }
  namep def-name namel cmove
  rr rl ( remaining string back on stack )

  \ Walk body words, tracking stack depth
  0 0 { depth mindep }
  begin
    next-word ( rem reml tok tokl )
    dup 0= if
      \ Ran out without ';'
      2drop 2drop false exit
    then
    \ Check for ';'
    over c@ [char] ; = over 1 = and if
      2drop 2drop
      mindep negate
      depth mindep -
      true exit
    then
    \ Look up word in dictionary
    2dup dict-find-i ( rem reml tok tokl entry|0 )
    dup if
      \ Found — apply stack effect
      nip nip dict-effect ( rem reml consumed produced )
      swap ( rem reml produced consumed )
      depth swap - to depth
      depth mindep min to mindep
      depth + to depth
    else
      drop ( rem reml tok tokl )
      \ Try as numeric literal
      2dup parse-number? if
        drop 2drop ( rem reml )
        depth 1+ to depth
      else ( rem reml tok tokl )
        ." unknown word " [char] " emit type [char] " emit cr
        2drop false exit
      then
    then
  again ;

\ validate-and-compile: validate, then add to dictionary on success.
: validate-and-compile ( c-addr u -- flag )
  2dup validate if ( c-addr u consumed produced )
    { con pro }
    2drop
    def-name def-name-len @ con pro dict-add
    ." OK: " def-name def-name-len @ type
    ."  ( " con 0 .r ."  -- " pro 0 .r ."  )" cr
    true
  else ( c-addr u )
    2drop false
  then ;

\ ============================================================
\ SEE — show word info
\ ============================================================

: dict-see ( c-addr u -- )
  dict-find-i dup 0= if drop ." word not found" cr exit then
  ." : " dup entry>name type ."   "
  ." ( " dup entry>consumed 0 .r ."  -- " entry>produced 0 .r ."  )" cr ;

\ ============================================================
\ FORGET — remove word from dictionary
\ ============================================================

: dict-forget ( c-addr u -- )
  2dup dict-hash cells dict-table + { bucket }
  bucket @
  dup 0= if drop 2drop ." not found" cr exit then
  dup entry>name 2over 2over istr= if
    2drop @ bucket ! 2drop exit
  then 2drop
  begin
    dup @ dup
  while
    dup entry>name 2over 2over istr= if
      2drop dup @ @ rot ! 2drop exit
    then 2drop
    nip @
  repeat
  2drop drop ." not found" cr ;
