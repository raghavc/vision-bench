"""Document reasoning over rendered receipts: OCR plus multi-step policy arithmetic.

The model must read line items (quantity, unit price, category tag), ignore
voided (struck-through) lines, apply category discounts and category-specific
tax exactly as the printed policy says, and round to cents. Higher levels
remove printed line totals (forcing multiplication), add voids, and render the
receipt slightly rotated.
"""

from __future__ import annotations

from decimal import ROUND_HALF_UP, Decimal

from visionbench.core.types import AnswerSpec
from visionbench.tasks.base import TaskFamily, register
from visionbench.tasks.render import canvas, finish, font

ITEMS = {
    "P": ["Apples", "Bananas", "Spinach", "Tomatoes", "Avocados", "Carrots", "Lemons", "Peppers"],
    "H": ["Dish soap", "Paper towels", "Sponges", "Trash bags", "Bleach", "Light bulbs", "Detergent"],
    "B": ["Sourdough", "Bagels", "Croissant", "Rye loaf", "Muffins", "Baguette"],
}
CAT_NAMES = {"P": "produce", "H": "household", "B": "bakery"}
#            d1 d2 d3 d4 d5
N_ITEMS = [0, 5, 7, 8, 10, 12]
LINE_TOTALS = [0, 1, 1, 1, 1, 0]
DISCOUNT = [0, 0, 1, 1, 1, 1]
TAX = [0, 0, 0, 1, 1, 1]
VOIDS = [0, 0, 0, 0, 1, 2]
ROTATE = [0, 0, 0, 0, 1, 1]
QUESTION = [None, "category_sum", "subtotal", "total", "total", "total"]

CENT = Decimal("0.01")


def money(x: Decimal) -> Decimal:
    return x.quantize(CENT, rounding=ROUND_HALF_UP)


@register
class ReceiptReasoning(TaskFamily):
    name = "receipt_reasoning"
    description = "Read a receipt image and compute amounts under printed discount/tax/void rules."
    skills = ["ocr", "document_understanding", "arithmetic", "instruction_following"]

    def build(self, rng, style, difficulty, noise_rng):
        lines = []
        cats = list(ITEMS)
        for _ in range(N_ITEMS[difficulty]):
            c = str(rng.choice(cats))
            name = str(rng.choice(ITEMS[c]))
            qty = int(rng.choice([1, 1, 1, 2, 2, 3, 4]))
            price = Decimal(int(rng.integers(49, 1299))) / 100
            lines.append({"cat": c, "name": name, "qty": qty, "price": price, "void": False})
        void_idx = rng.choice(len(lines), size=VOIDS[difficulty], replace=False) if VOIDS[difficulty] else []
        for i in void_idx:
            lines[int(i)]["void"] = True

        disc_cat = str(rng.choice(cats)) if DISCOUNT[difficulty] else None
        disc_pct = int(rng.choice([10, 15, 20, 25])) if disc_cat else 0
        tax_cat = "H" if TAX[difficulty] else None
        tax_pct = Decimal(str(rng.choice(["6.5", "7", "8", "8.25", "9.5"]))) if tax_cat else Decimal(0)

        live = [ln for ln in lines if not ln["void"]]
        cat_totals = {c: sum((ln["price"] * ln["qty"] for ln in live if ln["cat"] == c), Decimal(0)) for c in cats}
        subtotal = sum(cat_totals.values(), Decimal(0))
        discount = money(cat_totals[disc_cat] * disc_pct / 100) if disc_cat else Decimal(0)
        taxable = cat_totals[tax_cat] - (discount if disc_cat == tax_cat else 0) if tax_cat else Decimal(0)
        tax = money(taxable * tax_pct / 100) if tax_cat else Decimal(0)
        total = money(subtotal - discount + tax)

        q = QUESTION[difficulty]
        if q == "category_sum":
            c = str(rng.choice([c for c in cats if cat_totals[c] > 0]))
            value = money(cat_totals[c])
            question = f"What is the total amount spent on {CAT_NAMES[c]} items (category {c})?"
        elif q == "subtotal":
            value = money(subtotal)
            question = "What is the subtotal of all items, before any discount or tax?"
        else:
            value = total
            question = "Applying the store policy printed at the bottom of the receipt, what is the final amount due?"

        policy = []
        if disc_cat:
            policy.append(f"* {disc_pct}% off all {CAT_NAMES[disc_cat].upper()} ({disc_cat}) items")
        if tax_cat:
            policy.append(f"* Sales tax {tax_pct}% on HOUSEHOLD (H) items only,")
            policy.append("  charged after discounts")
        if policy:
            policy.append("* Discount and tax are each rounded")
            policy.append("  to the nearest cent")
        img = self._render(lines, policy, bool(LINE_TOTALS[difficulty]), style, rng)
        if ROTATE[difficulty]:
            img = img.rotate(float(rng.uniform(-4, 4)), expand=True, fillcolor=style.background)
        img = finish(img, style, noise_rng)
        prompt = ("The image shows a store receipt. Lines that are struck through were voided and are not charged. "
                  f"{question}")
        return {
            "prompt": prompt,
            "answer": AnswerSpec(kind="number", value=float(value), tolerance=0.005,
                                 format_hint="a dollar amount like 41.37 (no currency symbol needed)"),
            "images": [img],
            "facts": {
                "lines": [{**ln, "price": str(ln["price"])} for ln in lines],
                "category_totals": {c: str(v) for c, v in cat_totals.items()},
                "subtotal": str(money(subtotal)), "discount": str(discount), "tax": str(tax), "total": str(total),
            },
        }

    def _render(self, lines, policy, show_totals, style, rng):
        mono = font(22, "DejaVu Sans Mono")
        mono_b = font(26, "DejaVu Sans Mono", True)
        W = 640
        H = 260 + 40 * len(lines) + 34 * len(policy) + 60
        img, draw = canvas(W, H, style)
        paper = (252, 251, 246)
        draw.rectangle([20, 10, W - 20, H - 10], fill=paper, outline=(200, 200, 200))
        ink = style.ink
        draw.text((W / 2, 50), "GREENLEAF MARKET", font=mono_b, fill=ink, anchor="mm")
        draw.text((W / 2, 84), "Store #0417   Reg 03   10/14", font=mono, fill=ink, anchor="mm")
        y = 130
        draw.text((40, y), "QTY ITEM              CAT   PRICE" + ("   TOTAL" if show_totals else ""),
                  font=font(18, "DejaVu Sans Mono", True), fill=ink, anchor="lm")
        y += 16
        draw.line([(40, y), (W - 40, y)], fill=ink, width=1)
        y += 28
        for ln in lines:
            item = f"{ln['qty']:>2}  {ln['name']:<16}  {ln['cat']}  {ln['price']:>6}"
            if show_totals:
                item += f"  {ln['price'] * ln['qty']:>7.2f}"
            draw.text((40, y), item, font=font(19, "DejaVu Sans Mono"), fill=ink, anchor="lm")
            if ln["void"]:
                bbox = draw.textbbox((40, y), item, font=font(19, "DejaVu Sans Mono"), anchor="lm")
                draw.line([(bbox[0] - 4, y), (bbox[2] + 4, y)], fill=(40, 40, 40), width=3)
                draw.text((bbox[2] + 8, y), "VOID", font=font(15, "DejaVu Sans Mono", True), fill=(190, 30, 30),
                          anchor="lm")
            y += 40
        draw.line([(40, y), (W - 40, y)], fill=ink, width=1)
        y += 30
        draw.text((40, y), "Categories: P=produce H=household B=bakery", font=font(16, "DejaVu Sans Mono"),
                  fill=ink, anchor="lm")
        y += 40
        if policy:
            draw.text((40, y), "STORE POLICY", font=font(19, "DejaVu Sans Mono", True), fill=ink, anchor="lm")
            y += 32
            for p in policy:
                draw.text((40, y), p, font=font(17, "DejaVu Sans Mono"), fill=ink, anchor="lm")
                y += 30
        return img
