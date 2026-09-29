**GPU:** Tesla T4  **params:** 20.80M  **tokens/run:** 50M  **autocast dtype:** torch.float16

> Note: rows marked *re-run* had their loss measured in a long bf16-emulated run (loss is unaffected) and their tokens/s + peak memory re-measured in a short fp16 run of the identical configuration, because a T4 only emulates bf16.

**Max batch probe** (Tesla T4, 16 GB):

| model | max batch | peak GB |
|---|---|---|
| baseline | 496 | 14.81 |
| midpoint/no-rev | 512 | 15.20 |
| midpoint/rev | 2560 | 14.80 |

| run | mode | rev backprop | batch | steps | final train loss | final val loss | tokens/s | peak GB | minutes | speed/mem measured in |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline_B32 | baseline | no | 32 | 6103 | 1.8766 | 1.9611 | 35,331 | 4.43 | 80.4 | 300-step float16 re-run |
| hamiltonian_B32 | hamiltonian | yes | 32 | 6103 | 1.9150 | 2.0019 | 30,312 | 4.43 | 92.4 | 300-step float16 re-run |
| leapfrog_B32 | leapfrog | yes | 32 | 6103 | 1.8929 | 1.9774 | 29,140 | 4.43 | 100.5 | 300-step float16 re-run |
| midpoint_B32 | midpoint | yes | 32 | 6103 | 1.9052 | 1.9830 | 29,878 | 4.43 | 99.9 | 300-step float16 re-run |
| midpoint_B1920 | midpoint | yes | 1920 | 101 | 4.9837 | 4.9872 | 29,366 | 11.19 | 28.6 | same run |