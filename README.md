# CUDA_ASCON_DL

This CUDA Optimization of **ASCON** is used in the ToSC publication _Cryptanalysis: Theory versus Practice - Correcting Cryptanalysis Results on ASCON, ChaCha, and SERPENT using GPUs_ by Cihangir Tezcan, Gregor Leander, and Hosein Hadipour.

You can use benchmark to see how many 12-round ASCON initializations your GPU can perform in a second. These codes allowed us to perform 2^{35.10} ASCON initializations per second on an RTX 4090. 

We used these codes to experimentally verify the theoretically obtained 6-round differential-linear distinguisher on ASCON which has a claimed bias of 2^{-22.43}. We performed many experiments using 2^{55} data and concluded that the claimed distinguisher cannot be used as a distinguisher to distinguish 6 rounds of ASCON from a random permutation.
