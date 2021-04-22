#!/usr/bin/env bash

$SAGE -preparse common.sage
$SAGE -preparse starkad_poseidon.sage

mv common.sage.py common_sage.py
mv starkad_poseidon.sage.py starkad_poseidon_sage.py
