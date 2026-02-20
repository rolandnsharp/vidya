500000 constant MAX_TAPE
: hall allocate throw ;
0 constant OP_LEAF
variable tape-op   MAX_TAPE cells hall tape-op !
variable tape-arg0 MAX_TAPE cells hall tape-arg0 !
variable tape-arg1 MAX_TAPE cells hall tape-arg1 !
variable tape-data MAX_TAPE floats hall tape-data !
variable tape-grad MAX_TAPE floats hall tape-grad !
variable tape-len 0 tape-len !
: t-op   cells tape-op @ + ;
: t-data floats tape-data @ + ;
: t-grad floats tape-grad @ + ;
: v-leaf tape-len @ OP_LEAF over t-op ! dup t-data f! 0e over t-grad f! 1 tape-len +! ;

0.5e v-leaf .
1.0e v-leaf .
." tape-len=" tape-len @ .
0 t-data f@ f.
1 t-data f@ f. cr
bye
