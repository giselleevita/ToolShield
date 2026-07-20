# Split Hygiene Report

| Protocol | Check | Status | Detail |
|----------|-------|--------|--------|
| S_attack_holdout | template_leakage | PASS | No template leakage (train=15, val=5, test=6) |
| S_attack_holdout | af4_holdout | PASS | AF4 correctly held out (test has 125 AF4 samples) |
| S_attack_holdout | class_balance | PASS | train: benign=300, attack=281 |
| S_attack_holdout | class_balance | PASS | val: benign=100, attack=94 |
| S_random | template_leakage | PASS | No template leakage (train=15, val=5, test=6) |
| S_random | class_balance | PASS | train: benign=350, attack=251 |
| S_random | class_balance | PASS | val: benign=50, attack=125 |
