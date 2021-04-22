#!/usr/bin/env bash

cd rtt_mpc || exit 1

$SAGE -preparse common.sage
$SAGE -preparse starkad_poseidon.sage

mv common.sage.py common_sage.py
mv starkad_poseidon.sage.py starkad_poseidon_sage.py
