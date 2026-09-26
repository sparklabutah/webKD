# WebKD theory, machine-checked in Lean 4

Lean 4 + Mathlib formalization of the theoretical claims of the WebKD paper (`../main.tex`).
Every statement below is proved without `sorry`; `Audit.lean` prints the axioms each headline
result depends on (only `propext`, `Classical.choice`, `Quot.sound`).

## Build

```
lake exe cache get   # fetch the Mathlib cache once (a few minutes)
lake build           # builds all modules
lake env lean Audit.lean   # axiom audit of the headline theorems
```

Toolchain: `lean-toolchain` pins Lean `v4.34.0-rc2`, Mathlib at the matching tag.
