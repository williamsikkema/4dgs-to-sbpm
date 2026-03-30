# LED sync wand firmware

## Wiring

- **PIN_LED (D9):** connect to gate of N-channel MOSFET (e.g. IRLZ44N) with **logic-level** threshold, or NPN transistor (2N2222) with base resistor ~220Ω–1kΩ.
- **LED cathode** to MOSFET drain, **anode** through current-limiting resistor to `+V`.
- **PIN_BUTTON (D2):** one side to ground, other to pin (internal pull-up enabled) — short to GND to toggle mode.

## Current / resistor

For a single 5 mm LED at 20 mA and 5 V supply: `R ≈ (Vcc - Vf) / If` (e.g. `(5 - 2)/0.02 ≈ 150 Ω` for red). For high-power LEDs use appropriate **constant-current** drive or larger MOSFET + heatsink; do not exceed LED datasheet current.

## Modes

1. **Bright:** continuous `analogWrite(255)` for blob centroid tracking.
2. **Coded:** LFSR-driven blinking at ~`1/(2*TICK_US)` per bit transitions (see `TICK_US` in sketch).

## ESP32 / RP2040

- Replace `micros()` timing with hardware timer if you need jitter below ~10 µs.
- `analogWrite` may map to different PWM pins — set `PIN_LED` accordingly.

## Safety

This code does not limit current; **you** must size the resistor or CC supply. Overcurrent can destroy the LED or start a fire on high-power parts.
