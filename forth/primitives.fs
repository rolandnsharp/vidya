\ primitives.fs — Pre-populate dictionary with standard Forth words
\ Each entry: s" NAME" consumed produced dict-add

\ Stack manipulation
s" DUP"    1 2 dict-add      \ ( n -- n n )
s" DROP"   1 0 dict-add      \ ( n -- )
s" SWAP"   2 2 dict-add      \ ( a b -- b a )
s" OVER"   2 3 dict-add      \ ( a b -- a b a )
s" ROT"    3 3 dict-add      \ ( a b c -- b c a )
s" NIP"    2 1 dict-add      \ ( a b -- b )
s" TUCK"   2 3 dict-add      \ ( a b -- b a b )
s" 2DUP"   2 4 dict-add      \ ( a b -- a b a b )
s" 2DROP"  2 0 dict-add      \ ( a b -- )
s" 2SWAP"  4 4 dict-add      \ ( a b c d -- c d a b )
s" 2OVER"  4 6 dict-add      \ ( a b c d -- a b c d a b )

\ Arithmetic
s" +"      2 1 dict-add      \ ( a b -- sum )
s" -"      2 1 dict-add
s" *"      2 1 dict-add
s" /"      2 1 dict-add
s" MOD"    2 1 dict-add
s" /MOD"   2 2 dict-add      \ ( a b -- rem quot )
s" NEGATE" 1 1 dict-add
s" ABS"    1 1 dict-add
s" MIN"    2 1 dict-add
s" MAX"    2 1 dict-add
s" 1+"     1 1 dict-add
s" 1-"     1 1 dict-add
s" 2*"     1 1 dict-add
s" 2/"     1 1 dict-add

\ Comparison
s" ="      2 1 dict-add      \ ( a b -- flag )
s" <>"     2 1 dict-add
s" <"      2 1 dict-add
s" >"      2 1 dict-add
s" <="     2 1 dict-add
s" >="     2 1 dict-add
s" 0="     1 1 dict-add
s" 0<"     1 1 dict-add
s" 0>"     1 1 dict-add

\ Memory
s" @"      1 1 dict-add      \ ( addr -- value )
s" !"      2 0 dict-add      \ ( value addr -- )
s" C@"     1 1 dict-add
s" C!"     2 0 dict-add
s" +!"     2 0 dict-add      \ ( n addr -- )

\ Logic
s" AND"    2 1 dict-add
s" OR"     2 1 dict-add
s" XOR"    2 1 dict-add
s" INVERT" 1 1 dict-add

\ I/O
s" ."      1 0 dict-add      \ ( n -- )
s" EMIT"   1 0 dict-add
s" CR"     0 0 dict-add
s" SPACE"  0 0 dict-add
s" SPACES" 1 0 dict-add      \ ( n -- )
s" TYPE"   2 0 dict-add      \ ( addr u -- )
