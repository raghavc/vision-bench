"""Analog clock reading and elapsed-time reasoning (including mirror images).

Clock reading is a documented weakness of frontier VLMs. Both hands are drawn
at their physically correct angles (the hour hand advances with the minutes),
so the time can be recovered exactly. Higher levels drop numerals, add a
distracting second hand, and show the second clock as seen in a mirror, which
needs a reflection before the subtraction.
"""

from __future__ import annotations

import math

from visionbench.core.types import AnswerSpec
from visionbench.tasks.base import TaskFamily, register
from visionbench.tasks.render import canvas, finish

ROMAN = ["XII", "I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X", "XI"]
#                d1  d2  d3  d4  d5
NUMERALS = [None, "arabic", "arabic", None, "roman", None]
MINUTE_STEP = [0, 5, 1, 1, 1, 1]
SECOND_HAND = [0, 0, 0, 1, 1, 1]
MIRROR = [0, 0, 0, 0, 1, 1]


@register
class ClockReasoning(TaskFamily):
    name = "clock_reasoning"
    description = "Read analog clocks (incl. mirrored) and compute elapsed time."
    skills = ["perception", "measurement", "arithmetic"]

    def build(self, rng, style, difficulty, noise_rng):
        step = MINUTE_STEP[difficulty]
        while True:
            t1 = int(rng.integers(0, 720 // step)) * step
            t2 = int(rng.integers(0, 720 // step)) * step
            elapsed = (t2 - t1) % 720
            # avoid hands overlapping exactly (hides one hand) and trivial elapsed times
            if 30 <= elapsed <= 690 and not self._overlap(t1) and not self._overlap(t2):
                break
        mirror = bool(MIRROR[difficulty])
        W, H = 1000, 520
        img, draw = canvas(W, H, style)
        for i, (t, mir) in enumerate(((t1, False), (t2, mirror))):
            cx, cy = 250 + i * 500, 270
            self._clock(draw, cx, cy, 200, t, NUMERALS[difficulty], bool(SECOND_HAND[difficulty]), mir, rng, style)
            label = "Start" if i == 0 else ("End (seen in a mirror)" if mir else "End")
            draw.text((cx, 30), label, font=style.font(24, bold=True), fill=style.ink, anchor="mm")
        img = finish(img, style, noise_rng)
        prompt = ("The left clock shows when a task started and the right clock shows when it ended, later the same "
                  "day; less than 12 hours passed. "
                  + ("The right clock is shown as it appears in a mirror (left and right are swapped). " if mirror else "")
                  + ("Ignore the thin red second hands. " if SECOND_HAND[difficulty] else "")
                  + "How many minutes did the task take?")
        return {
            "prompt": prompt,
            "answer": AnswerSpec(kind="integer", value=elapsed, tolerance=0,
                                 format_hint="a whole number of minutes"),
            "images": [img],
            "facts": {"start_hhmm": self._fmt(t1), "end_hhmm": self._fmt(t2), "elapsed_min": elapsed,
                      "mirrored_end": mirror},
        }

    @staticmethod
    def _fmt(t):
        h, m = divmod(t, 60)
        return f"{h if h else 12}:{m:02d}"

    @staticmethod
    def _overlap(t):
        h, m = divmod(t, 60)
        hour_deg = (h % 12) * 30 + m * 0.5
        min_deg = m * 6
        return abs((hour_deg - min_deg + 180) % 360 - 180) < 8

    def _clock(self, draw, cx, cy, R, t, numerals, second, mirror, rng, style):
        ink = style.ink
        sgn = -1 if mirror else 1

        def pt(deg, r):
            a = math.radians(deg)
            return cx + sgn * r * math.sin(a), cy - r * math.cos(a)

        draw.ellipse([cx - R, cy - R, cx + R, cy + R], fill=(252, 252, 250), outline=ink, width=style.lw(6))
        for k in range(60):
            r0 = R - (24 if k % 5 == 0 else 12)
            draw.line([pt(k * 6, r0), pt(k * 6, R - 4)], fill=ink, width=4 if k % 5 == 0 else 2)
        if numerals:
            f = style.font(30 if numerals == "arabic" else 24, bold=True)
            for k in range(12):
                txt = str(k if k else 12) if numerals == "arabic" else ROMAN[k]
                x, y = pt(k * 30, R - 50)
                # I, V and X are mirror-symmetric glyphs, so a mirrored Roman numeral is just its letters reversed
                draw.text((x, y), txt[::-1] if mirror and numerals == "roman" else txt, font=f, fill=ink, anchor="mm")
        h, m = divmod(t, 60)
        hour_deg = (h % 12) * 30 + m * 0.5
        min_deg = m * 6
        draw.line([(cx, cy), pt(hour_deg, R * 0.5)], fill=ink, width=style.lw(12))
        draw.line([(cx, cy), pt(min_deg, R * 0.82)], fill=ink, width=style.lw(7))
        if second:
            sdeg = float(rng.integers(0, 60)) * 6
            draw.line([pt(sdeg + 180, R * 0.15), pt(sdeg, R * 0.9)], fill=(210, 40, 40), width=2)
        draw.ellipse([cx - 10, cy - 10, cx + 10, cy + 10], fill=ink)
