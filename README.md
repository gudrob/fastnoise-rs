# fast-noise-simd-rs

[![CI](https://github.com/gudrob/fastnoise-rs/actions/workflows/ci.yml/badge.svg)](https://github.com/gudrob/fastnoise-rs/actions/workflows/ci.yml)

Zero-dependency Rust port of [Auburn/FastNoiseSIMD](https://github.com/Auburn/FastNoiseSIMD).
Generiert kohärentes Rauschen für prozedurale Content-Generierung.

Build: `cargo build --release`
Test:  `cargo test`
Bench: `cargo bench`

---

## Architektur

Alle Noise-Funktionen arbeiten im `VECTOR_SIZE` breiten Batch-Modus.
`VECTOR_SIZE` wird vom SIMD-Backend bestimmt:
  Scalar=1, SSE2/SSE41/NEON=4, AVX2=8, AVX512=16

Dispatching bei `FastNoise::new()` via `std::is_x86_feature_detected!` / `std::is_aarch64_feature_detected!`.
Zur Laufzeit wird eine Funktionstabelle (`NoiseKernel`) mit SIMD-gecasteten Funktionspointern aufgebaut.

### Datei-Übersicht

```
src/
  lib.rs          — FastNoise facade, Builder-Pattern, Grid-Generatoren, SIMD-Dispatch
  settings.rs     — Settings struct (NoiseType, FractalType, Cellular*, Perturb*, Achsen-Skalen). Kein SIMD.
  hash.rs         — xorshift32 PRNG, val_coord_f32 (deterministischer Hash aus Koordinaten).
                      Lookup-Tabelle aus `FN_DECIMAL` für Integer→Float-Konversion.
  vectorset.rs    — Gradient-Vektor-Tabelle (24 Vektoren) für Perlin/Simplex.
  kernel.rs       — Batched SIMD-Kernel: smoothstep, lerp, koordinaten-batching, Grid-Fills.
  noise.rs        — Pro Noise-Type eine Funktion (value_2d/3d, perlin_2d/3d, simplex_2d/3d etc.).
                      Alle arbeiten mit generischen F: SimdFloat + I: SimdInt.
  fractal.rs      — fBm, Billow, RigidMulti. Octaven-Schleife in SIMD.
  perturb.rs      — Domain Warping: sampler-Ergebnis als Input für weiteren Noise-Call.
  simd/
    mod.rs        — Traits: SimdFloat, SimdInt. Methoden: set, mul_add, floor, blend, cmp_gt etc.
    scalar.rs     — 1-Lane Fallback. Alle Operationen als einfache f32/i32-Wrapper.
    sse2.rs       — 4-Lane x86_64 SSE2 (baseline auf amd64).
    sse41.rs      — 4-Lane SSE4.1 (floor, blend, i32-mul via _mm_mullo_epi32).
    avx2.rs       — 8-Lane AVX2 + FMA (256-bit).
    avx512.rs     — 16-Lane AVX-512F (512-bit). Blend via Masks, floor via _mm512_roundscale_ps.
    neon.rs       — 4-Lane ARM NEON (aarch64).
benches/
  noise_benchmarks.rs — Criterion-Benches für single-point 2D/3D + Grid-Generation.
```

### Datenfluss single-point (get_noise_3d)

```
Settings (Frequenz, Achsen-Skalen, Seed)
  → Skalierte Koordinaten (x*FREQ * X_SCALE)
  → Noise-Funktion aus noise.rs (generisch über SimdFloat)
      → Cell-Indizes per floor()
      → Hash pro Lane per hash.rs
      → Interpolation (Hermite/Lerp/Bicubic)
  → Fractal-Overlay (octaves-Schleife)
  → Perturb (rekursiver Noise-Call mit perturbierter Koordinate)
  → [0..1] clamp
```

### Merge-Konflikt Doku (kurz)

Stand nach 23 commits ab HEAD. Branch `simd-batch-kernel` hat `kernel.rs` introziert
(alle noise-Funktionen von `noise.rs` nach `kernel.rs` portiert aber inkomplett integriert).
`lib.rs` Grid-Generatoren rufen noch `noise.rs` Funktionen. Nächster Schritt:

  1. `lib.rs` Grid-Generatoren auf `kernel.rs` Funktionen umstellen
  2. `noise.rs` obsolet machen (entweder löschen oder als legacy behalten)
  3. SIMD-Dispatch so umbauen dass `kernel.rs` generics über Box<dyn Trait> statt Monomorphisation.

---

## TODOs

### Kritisch (blocker für 0.1 release)

- [x] `lib.rs` Grid-Generatoren (`generate_grid_2d`/`3d`) von `noise::value_2d` auf
      `kernel::fill_noise_set_3d`/`fill_noise_set_2d` umgestellt.
      Value/Perlin/Simplex nutzen SIMD-Kernel, rest per scalar fallback.
- [x] `kernel.rs` simplex_3d batch per scalar fallback (per-lane evaluation)
      repariert — original stub durch echte Delegation an `single_simplex_3d` ersetzt.
- [x] `hash_batch_3d_x` hat seed-Parameter
- [x] `noise_generate_sample_3d` in `kernel.rs` delegiert nun an `noise::generate_3d`
      statt duplicate dispatch-Logik.
- [x] `build.rs` / Feature-Gates geprüft: Cargo Features (`sse2`, `sse41`, `avx2`, `avx512`, `neon`)
      werden via `build.rs` korrekt auf `has_*` cfg-Flags gemappt. Automatische Erkennung via
      `target_feature` als Fallback. `rustc-check-cfg` unterdrückt Warnings für custom cfgs.

### Wichtig (0.1 polish)

- [ ] Test gegen C-Referenzwerte (golden file):
      FastNoiseSIMD mit seed 1337, bekannte Koordinaten, Werte in Datei speichern,
      gegen Rust-Output diffen. Ohne das keine Garantie auf Bit-Identität.
- [ ] Benchmarks mit SIMD-Backends laufen lassen (nicht nur scalar).
      Aktuell alle benches scalar weil kernel.rs dispatch fehlt.
- [ ] `kernel.rs` `#[allow(dead_code)]` aufräumen nach Grid-Integration.
- [ ] Frequenz-Berechnung in fill_* Funktionen prüfen:
      aktuelle freq Skalierung matcht nicht exakt FastNoiseSIMD (dort: `x * m_frequency`).
- [ ] Perturb in kernel.rs portieren: aktuell nur noise.rs/perturb.rs.
- [ ] Error-Handling: Settings-Validierung (Frequenz > 0, Octaves > 0, etc.)

### Optional / Nice-to-have

- [ ] Cellular Return-Types implementieren die noch fehlen (NoiseLookup, Distance2Cave, etc.)
- [ ] SIMD-Trait um sqrt/rsqrt erweitern (aktuell per-Lane scalar `f32::sqrt`).
- [ ] NEON-Tests auf echter ARM-Hardware (aktuell nur compiliert, nie getestet).
- [ ] AVX-512 Tests auf Hardware mit AVX-512F (z.B. Skylake-X, Ice Lake).
- [ ] C-FFI Schnittstelle (extern "C" fn), kompatibel zur originalen FastNoiseSIMD C-API.
- [ ] `#[no_std]` support (nur alloc für Vec, keine std-Deps).

---

## SIMD Backend Details

### Trait-Methoden

```rust
pub trait SimdFloat: Copy {
    const VECTOR_SIZE: usize;
    unsafe fn load(ptr: *const f32) -> Self;
    unsafe fn store(self, ptr: *mut f32);
    fn set(val: f32) -> Self;
    fn zero() -> Self;
    fn mul(self, rhs: Self) -> Self;
    fn mul_add(self, a: Self, b: Self) -> Self;  // self*a + b (FMA)
    fn add(self, rhs: Self) -> Self;
    fn sub(self, rhs: Self) -> Self;
    fn floor(self) -> Self;
    fn blend(self, other: Self, mask: Self) -> Self;
    fn cmp_gt(self, other: Self) -> Self;
    fn to_int_trunc() -> ???  // via companion SimdInt trait
}
pub trait SimdInt: Copy {
    const VECTOR_SIZE: usize;
    unsafe fn load(ptr: *const i32) -> Self;
    fn set(val: i32) -> Self;
    fn add(self, rhs: Self) -> Self;
    fn sub(self, rhs: Self) -> Self;
    fn mul(self, rhs: Self) -> Self;
    fn to_float<F: SimdFloat>(self) -> F;
    fn cast_f32_to_i32_trunc<F: SimdFloat>(val: F) -> Self;
    fn and(self, rhs: Self) -> Self;
    fn or(self, rhs: Self) -> Self;
    fn xor(self, rhs: Self) -> Self;
    fn shift_left(self, rhs: Self) -> Self;
    fn shift_right_arithmetic(self, rhs: Self) -> Self;
}
```

### Backend-Auswahl (build.rs)

`build.rs` mappt Cargo Features (`sse2`, `sse41`, `avx2`, `avx512`, `neon`) auf
`has_*` cfg-Flags. Zusätzlich automatische Erkennung via `target_feature` vom Compiler.
`rustc-check-cfg` unterdrückt Warnings für die custom cfgs.

Ohne Feature-Flag: nur Scalar. Ohne passendes `target_feature`: nur Scalar.
SIMD-Module werden nur mit aktivem Backend compiliert – sonst stub (alias auf Scalar).

Vector-Lane-Zuordnung:
  SSE2:     __m128 (4xf32), __m128i (4xi32)
  SSE4.1:   dito, aber floor via _mm_floor_ps, i32-mul via _mm_mullo_epi32
  AVX2:     __m256 (8xf32), __m256i (8xi32)
  AVX-512F: __m512 (16xf32), __m512i (16xi32)
  NEON:     float32x4_t (4xf32), int32x4_t (4xi32)

---

## Hash-Formeln

Alle Hash-Formeln 1:1 aus FastNoiseSIMD `FastNoiseSIMD_internal.cpp` portiert.

```
val_coord(seed, x)            = x XOR seed, dann xorshift32, dann * LOOKUP
val_coord_2d(seed, x, y)      = val_coord(seed, x + y * PRIME_X)
val_coord_3d(seed, x, y, z)   = val_coord(seed, x + y*PRIME_X + z*PRIME_Y)
grad_coord_2d(seed, x, y, xi, yi) = Gradienten-Dotprodukt über 2D-Gradienten-Tabelle
grad_coord_3d(seed, x, y, z, xi, yi, zi) = dito für 3D
```

`PRIME_X = 501125321`, `PRIME_Y = 1136930381`, `PRIME_Z = 1720413743`
`LOOKUP_SIZE = 2048`, `LOOKUP_HALF = 1024`

Gradienten-Tabelle: 24 3D-Einheitsvektoren (gleiche wie in FastNoiseSIMD, ohne eine Ecke vom Würfel),
als `[f32; 72]` Array (24 Vektoren × 3 Komponenten).

---

## Warum nur scalar-Benchmarks aktuell?

Die Noise-Kernel sind auf generische `F: SimdFloat + I: SimdInt` ausgelegt,
aber das SIMD-Dispatching in `lib.rs` passiert über eine statische Dispatch-Tabelle
die noch nicht auf die batched-Funktionen umgestellt ist.
Aktuell läuft alles durch `noise.rs` generics, und der Compiler monomorphisiert
automatisch auf das zur Laufzeit erkannte SIMD-Backend (theoretisch).
Praktisch wurde das noch nicht verifiziert weil die `kernel.rs` Integration nicht
abgeschlossen ist. Daher benchen wir gegen scalar-backend als Baseline.

---

## Lizenz

MIT — siehe LICENSE Datei.