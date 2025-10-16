# Theme & Stage Visuals — Infant LLM v6.3

This UI adapts visually to the model’s maturity. All changes are subtle and silent (no sounds/popups).

## Stage → Visual Mapping

| Stage     | Input/Console Accent | Thinking Pulse Cycle | Dimming While Thinking |
|-----------|-----------------------|----------------------|------------------------|
| Infant    | Green (#34d399)       | 6s (fastest)         | 0.7 (light)            |
| Child     | Blue (#60a5fa)        | 8s                   | 0.65                   |
| Teenager  | Purple (#a78bfa)      | 10s                  | 0.6                    |
| Adult     | Gold (#fbbf24)        | 12s (calmest)        | 0.5 (deep focus)       |

- **Ambient Pulse**: `body.thinking.stage { animation: thinkingPulse … }`
- **Chat Dimming**: `body.thinking.stage #chat-output { opacity: … }`
- **Accent Borders**: `body.stage #chat-output`, `body.stage #user-input`
- **Stage Text Glow**: `.stage-text` changes color/glow while thinking

## Learning Flash Colors (adaptive)
- General: green
- Math: blue
- Science: purple
- Technology: gold
- Engineering: coral

Edit in `style.css` and `infant_ui_v6_3.html`:
- CSS selectors for stage tones
- JS `triggerLearningFlash(category)` mapping

> Keep everything subtle. The UI should feel alive — not noisy.
