/**
 * LED synchronization wand — Arduino-compatible (AVR / ESP32 / RP2040 adaptable).
 *
 * Sync idea: a fast-moving bright point is visible in all fixed cameras; temporal
 * offset estimation correlates 1D signals (u,v,brightness) per view. Two modes:
 *   MODE_BRIGHT: steady LED for centroid tracking.
 *   MODE_CODED: pseudo-random on/off pattern (LFSR) for robust cross-correlation.
 *
 * Timing constants below are documented for matching expected camera exposure.
 * Use a current-limited driver (resistor or constant-current) for the LED.
 */

const int PIN_LED = 9;
const int PIN_BUTTON = 2;

// Mode A: full brightness (PWM 255 on 8-bit)
// Mode B: bit period in microseconds (toggle rate ~ 1/(2*TICK_US) Hz per channel)
const unsigned long TICK_US = 500;

enum Mode { MODE_BRIGHT = 0, MODE_CODED = 1 };
volatile Mode g_mode = MODE_BRIGHT;

// 16-bit LFSR for maximal length sequence (poly x^16+x^14+x^13+x^11)
uint16_t lfsr = 0xACE1u;

unsigned long lastTick = 0;
bool ledState = false;
bool lastButton = HIGH;
unsigned long debounceMs = 0;

void setup() {
  pinMode(PIN_LED, OUTPUT);
  pinMode(PIN_BUTTON, INPUT_PULLUP);
  digitalWrite(PIN_LED, LOW);
}

void loop() {
  unsigned long now = micros();

  // Debounced mode toggle
  bool b = digitalRead(PIN_BUTTON);
  if (b == LOW && lastButton == HIGH && (millis() - debounceMs) > 200) {
    g_mode = (g_mode == MODE_BRIGHT) ? MODE_CODED : MODE_BRIGHT;
    debounceMs = millis();
  }
  lastButton = b;

  if (g_mode == MODE_BRIGHT) {
    analogWrite(PIN_LED, 255);
    return;
  }

  // Coded mode: non-blocking bit clock
  if (now - lastTick >= TICK_US) {
    lastTick = now;
    unsigned lsb = lfsr & 1u;
    lfsr >>= 1;
    if (lsb)
      lfsr ^= 0xB400u;
    ledState = (lfsr & 1u) != 0;
    digitalWrite(PIN_LED, ledState ? HIGH : LOW);
  }
}
