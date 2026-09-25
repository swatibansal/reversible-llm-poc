**GPU:** cpu  **dtype:** torch.float32  **params:** 0.86M  **tokens/run:** 0M

| run | mode | rev backprop | batch | steps | final train loss | final val loss | tokens/s | peak GB | minutes |
|---|---|---|---|---|---|---|---|---|---|
| baseline_B16 | baseline | no | 16 | 195 | 3.2174 | 3.2349 | 29,460 | n/a | 0.1 |
| hamiltonian_B16 | hamiltonian | yes | 16 | 195 | 3.0933 | 3.1153 | 17,291 | n/a | 0.2 |
| leapfrog_B16 | leapfrog | yes | 16 | 195 | 3.1499 | 3.1676 | 17,097 | n/a | 0.2 |
| midpoint_B16 | midpoint | yes | 16 | 195 | 3.1936 | 3.2062 | 17,160 | n/a | 0.2 |
| midpoint_B64 | midpoint | yes | 64 | 48 | 4.7508 | 4.5789 | 18,048 | n/a | 0.2 |