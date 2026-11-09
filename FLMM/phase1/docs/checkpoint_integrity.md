# Checkpoint integrity — PASS

All 529 DeepSeekVL and 345 SAM image-encoder missing keys were compared to pinned pretrained disk tensors and verified unchanged after loading F-LMM. Every trained checkpoint tensor was verified equal after load. See results/checkpoint_integrity.json for per-key proof.

Old smoke did not record a backbone revision; historical revision remains UNVERIFIED. New pinned revision reproduces its 20 saved prediction masks exactly.
