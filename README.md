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
Zur Laufzeit wird via `simd_dispatch!`-Makro die richtige monomorphisierte `kernel::fill_noise_set_*`-Funktion aufgerufen.

### Datei-Übersicht

```
src/
  lib.rs          — FastNoise facade, Builder-Pattern, Grid-Generatoren, SIMD-Dispatch
  settings.rs     — Settings struct (NoiseType, FractalType, Cellular*, Perturb*, Achsen-Skalen). Kein SIMD.
  hash.rs         — xorshift32 PRNG, val_coord_f32 (deterministischer Hash aus Koordinaten).
                      Lookup-Tabelle aus `FN_DECIMAL` für Integer→Float-Konversion.
  vectorset.rs    — Gradient-Vektor-Tabelle (24 Vektoren) für Perlin/Simplex.
  kernel.rs       — Batched SIMD-Kernel: smoothstep, lerp, Grid-Fills,
                      Batch-Funktionen für Value/Perlin/Simplex + per-lane-Fallback.
  noise.rs        — Single-sample Noise-Funktionen (value_2d/3d, perlin_2d/3d, simplex_2d/3d etc.).
                      Generisch über F: SimdFloat + I: SimdInt.
  fractal.rs      — fBm, Billow, RigidMulti. Octaven-Schleife in SIMD.
  perturb.rs      — Domain Warping: sampler-Ergebnis als Input für weiteren Noise-Call.
  simd/
    mod.rs        — Traits: SimdFloat, SimdInt. Methoden: set, mul_add, floor, blend, cmp_gt etc.
    scalar.rs     — 1-Lane Fallback. Alle Operationen als f32/i32-Wrapper.
    sse2.rs       — 4-Lane x86_64 SSE2 (baseline auf amd64).
    sse41.rs      — 4-Lane SSE4.1 (floor via _mm_floor_ps, i32-mul via _mm_mullo_epi32).
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
  → noise::generate_3d (generisch über SimdFloat)
      → Cell-Indizes per floor()
      → Hash per hash.rs
      → Interpolation (Hermite/Lerp/Bicubic)
  → Fractal-Overlay (octaves-Schleife)
  → Perturb (rekursiver Noise-Call mit perturbierter Koordinate)
  → f32 return
```

### Datenfluss Grid (generate_grid)

```
FastNoise::generate_grid(start_x, start_y, start_z, w, h, d)
  → simd_dispatch!(level, fill_noise_set_3d, settings, …)
      → kernel::fill_noise_set_3d::<SIMD_FLOAT, SIMD_INT>
          → Pro y,z-Iteration: VECTOR_SIZE x-Werte per Batch (Value/Perlin/Simplex)
              → Fallback: per-lane via noise::generate_3d (Fractal/Cellular/Cubic/WhiteNoise)
```

---

## TODOs

### Kritisch / offen

- [ ] **Test gegen C-Referenzwerte (golden file):**
      FastNoiseSIMD mit seed 1337, bekannte Koordinaten, Werte in Datei speichern,
      gegen Rust-Output diffen. Ohne das keine Garantie auf Bit-Identität.
- [ ] **Benchmarks mit SIMD-Backends laufen lassen** (nicht nur scalar).
      Aktuell benchen alle benches im Scalar-Modus. Benchmarks müssen für alle
      Backends echte SIMD-Pfade durchlaufen.
- [ ] **Perturb in kernel.rs integrieren:**
      Perturb fällt in `kernel::noise_batch_3d` noch auf per-lane-Fallback zurück.
      Eine echte SIMD-Batch-Implementierung fehlt.
- [ ] **Frequenz-Skalierung prüfen:**
      Aktuelle freq-Skalierung matcht nicht exakt FastNoiseSIMD (dort: `x * m_frequency`).
      Code verwendet `x * x_scale * frequency`, was eine zusätzliche Multiplikation ist.
- [ ] **Error-Handling:** Settings-Validierung (Frequenz > 0, Octaves > 0, etc.)

### Optional / Nice-to-have

- [ ] Cellular Return-Types implementieren die noch fehlen (werden von `CellularReturnType` enum
      zwar definiert, aber nicht alle sind getestet/validiert).
- [ ] SIMD-Trait um sqrt/rsqrt erweitern (aktuell per-Lane scalar `f32::sqrt`).
- [ ] NEON-Tests auf echter ARM-Hardware (aktuell nur compiliert, nie getestet).
- [ ] AVX-512 Tests auf Hardware mit AVX-512F (z.B. Skylake-X, Ice Lake).
- [ ] C-FFI Schnittstelle (`extern "C" fn`), kompatibel zur originalen FastNoiseSIMD C-API.
- [ ] `#[no_std]` support (nur alloc für Vec, keine std-Deps).
- [ ] `noise.rs` und `kernel.rs` zusammenführen: `noise.rs` enthält die Single-Sample-Funktionen,
      `kernel.rs` die Batch-Kernel. Langfristig könnten die Batch-Kernel die Single-Sample
      komplett ersetzen.

### Erledigt

- [x] Grid-Generatoren (`generate_grid`/`generate_grid_2d`) dispatch-en via `simd_dispatch!`
      auf `kernel::fill_noise_set_3d`/`fill_noise_set_2d`.
      Value/Perlin/Simplex nutzen SIMD-Kernel, Rest per scalar fallback.
- [x] `kernel.rs` simplex_3d batch per scalar fallback (per-lane evaluation).
- [x] `hash_batch_3d_x` totcode entfernt.
- [x] `noise.rs` tote Konstanten `F3`/`G3` mit `#[allow(dead_code)]` entfernt.
- [x] `build.rs` / Feature-Gates geprüft: Cargo Features (`sse2`, `sse41`, `avx2`, `avx512`, `neon`)
      via `build.rs` korrekt auf `has_*` cfg-Flags gemappt. Automatische Erkennung via
      `target_feature` als Fallback. `rustc-check-cfg` unterdrückt Warnings.

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

`build.rs` mapped Cargo Features (`sse2`, `sse41`, `avx2`, `avx512`, `neon`) auf
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

## Lizenz

MIT — siehe LICENSE Datei.