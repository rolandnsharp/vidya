\ test.fs — Test cases for the symbolic stack effect validator

variable test-count
variable pass-count
variable fail-count
0 test-count !  0 pass-count !  0 fail-count !

: test-pass ( -- ) 1 pass-count +! 1 test-count +! ;
: test-fail ( c-addr u -- ) 1 fail-count +! 1 test-count +!
  ." FAIL: " type cr ;

: expect-true ( flag c-addr u -- )
  rot if 2drop test-pass else test-fail then ;

: expect-false ( flag c-addr u -- )
  rot 0= if 2drop test-pass else test-fail then ;

\ Test that validate returns specific stack effect
: expect-effect ( c-addr u expected-con expected-pro -- )
  { econ epro }
  2dup validate if
    { con pro }
    con econ = pro epro = and if
      2drop test-pass
    else
      test-fail
      ."   expected ( " econ . ." -- " epro . ." ) got ( " con . ." -- " pro . ." )" cr
    then
  else
    test-fail
    ."   expected valid but got rejected" cr
  then ;

\ ============================================================
cr ." === Valid definitions ===" cr
\ ============================================================

s" : SQUARE DUP * ;" validate-and-compile
s" SQUARE should compile" expect-true

s" : CUBE DUP SQUARE * ;" validate-and-compile
s" CUBE should compile (uses SQUARE)" expect-true

s" : DOUBLE DUP + ;" validate-and-compile
s" DOUBLE should compile" expect-true

s" : QUADRUPLE DOUBLE DOUBLE ;" validate-and-compile
s" QUADRUPLE should compile (uses DOUBLE)" expect-true

s" : NOOP ;" validate-and-compile
s" NOOP should compile (empty body)" expect-true

s" : ADD3 + + ;" validate-and-compile
s" ADD3 should compile (needs 3 inputs)" expect-true

s" : ZERO DROP 0 ;" validate-and-compile
s" ZERO should compile (number literal)" expect-true

s" : DOUBLE-NEGATE NEGATE NEGATE ;" validate-and-compile
s" DOUBLE-NEGATE should compile" expect-true

s" : SUM-OF-SQUARES DUP * SWAP DUP * + ;" validate-and-compile
s" SUM-OF-SQUARES should compile" expect-true

\ ============================================================
cr ." === Stack effect verification ===" cr
\ ============================================================

s" : T1 DUP * ;" 1 1 expect-effect           \ SQUARE: ( 1 -- 1 )
s" : T2 DUP + ;" 1 1 expect-effect           \ DOUBLE: ( 1 -- 1 )
s" : T3 ;" 0 0 expect-effect                  \ NOOP:   ( 0 -- 0 )
s" : T4 + + ;" 3 1 expect-effect              \ ADD3:   ( 3 -- 1 )
s" : T5 DROP 0 ;" 1 1 expect-effect           \ ZERO:   ( 1 -- 1 )
s" : T6 SWAP ;" 2 2 expect-effect             \ swap:   ( 2 -- 2 )
s" : T7 DUP DUP ;" 1 3 expect-effect         \ tripl:  ( 1 -- 3 )
s" : T8 DROP DROP ;" 2 0 expect-effect        \ drop2:  ( 2 -- 0 )
s" : T9 OVER + ;" 2 2 expect-effect           \ ( 2 -- 2 )
s" : T10 DUP ROT ;" 3 4 expect-effect         \ ( 3 -- 4 )

\ ============================================================
cr ." === Invalid definitions ===" cr
\ ============================================================

s" : BAD NONEXISTENT ;" validate-and-compile
s" unknown word should be rejected" expect-false

s" : BAD ALSONOTAWORD ;" validate-and-compile
s" another unknown word rejected" expect-false

s" : BAD DUP NOWORD * ;" validate-and-compile
s" unknown in middle rejected" expect-false

\ Missing semicolon
s" : BAD DUP *" validate-and-compile
s" missing semicolon rejected" expect-false

\ Missing name
s" : ;" validate-and-compile
s" missing name rejected" expect-false

\ ============================================================
cr ." === Compositional growth ===" cr
\ ============================================================

s" : SENSOR-READ @ ;" validate-and-compile
s" SENSOR-READ should compile" expect-true

s" : SENSOR-WRITE ! ;" validate-and-compile
s" SENSOR-WRITE should compile" expect-true

s" : SENSOR-AVG SENSOR-READ SWAP SENSOR-READ + 2 / ;" validate-and-compile
s" SENSOR-AVG should compile (builds on SENSOR-READ)" expect-true

s" : CLAMP ROT ROT MAX SWAP MIN ;" validate-and-compile
s" CLAMP should compile" expect-true

s" : ABS-DIFF - ABS ;" validate-and-compile
s" ABS-DIFF should compile" expect-true

s" : WITHIN OVER - ROT ROT - > ;" validate-and-compile
s" WITHIN should compile" expect-true

\ ============================================================
cr ." === Case insensitivity ===" cr
\ ============================================================

s" : LOWER-TEST dup * ;" validate-and-compile
s" lowercase dup should match DUP" expect-true

s" : MIXED-TEST Dup Swap ;" validate-and-compile
s" mixed case should work" expect-true

\ ============================================================
cr ." === Number literals ===" cr
\ ============================================================

s" : CONST42 42 ;" validate-and-compile
s" number literal 42" expect-true

s" : NEG-ONE -1 ;" validate-and-compile
s" negative number literal" expect-true

s" : ADD10 10 + ;" validate-and-compile
s" number + arithmetic" expect-true

s" : SCALE 100 * 50 + ;" validate-and-compile
s" multiple number literals" expect-true

s" : CONST42-FX ;" 0 0 expect-effect          \ re-check: ( 0 -- 0 )
s" : FX-ADD10 10 + ;" 1 1 expect-effect        \ ( 1 -- 1 )
s" : FX-SCALE 100 * 50 + ;" 1 1 expect-effect  \ ( 1 -- 1 )
s" : FX-CONST 42 ;" 0 1 expect-effect          \ ( 0 -- 1 )

\ ============================================================
cr ." === Results ===" cr
\ ============================================================

pass-count @ 0 .r ."  passed, " fail-count @ 0 .r ."  failed, "
test-count @ 0 .r ."  total" cr

fail-count @ 0<> if
  ." SOME TESTS FAILED" cr
  1 (bye)
else
  ." ALL TESTS PASSED" cr
then

bye
