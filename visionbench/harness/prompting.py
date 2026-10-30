"""Prompts for subjects. Kept deliberately neutral and identical across models."""

from __future__ import annotations

from visionbench.core.types import TaskInstance

SYSTEM_DIRECT = (
    "You are taking a visual reasoning test. Look at the image(s) carefully; details matter. "
    "Answers are graded automatically and exactly."
)

SYSTEM_AGENT = (
    "You are taking a visual reasoning test with tools. Look at the image(s) carefully; details matter. "
    "You can inspect regions of an image at full resolution with the `zoom` tool{code}. "
    "When you are done, call `submit_answer` exactly once with your final answer and your confidence. "
    "Answers are graded automatically and exactly."
)

ANSWER_PROTOCOL = (
    "\n\nBefore answering, briefly state the key observations from the image that your answer depends on. "
    "Then give your final answer inside <answer></answer> tags, formatted as: {fmt}. "
    "Finally, give the probability (0-100) that your answer is exactly correct inside <confidence></confidence> tags."
)

AGENT_PROTOCOL = (
    "\n\nFinal answer format (for submit_answer): {fmt}."
)

BLIND_PREFIX = (
    "[The image for this question is not available to you. Answer anyway with your single best guess; "
    "you must still give a final answer in the required format.]\n\n"
)

CAPTION_PROMPT = (
    "Describe this image exhaustively and precisely so that someone who cannot see it could answer detailed "
    "questions about it: list every object, label, number, colour, position, connection and spatial relationship "
    "you can see. Use coordinates or relative positions where helpful. Do not speculate about what the question "
    "might be."
)

CAPTION_PREFIX = (
    "[You cannot see the image. Instead, here is a detailed description of it written by another observer:]\n"
    "<image_description>\n{caption}\n</image_description>\n\n"
)


def user_text(inst: TaskInstance, mode: str, condition: str, caption: str | None = None) -> str:
    q = inst.prompt
    if condition == "blind":
        q = BLIND_PREFIX + q
    elif condition == "caption":
        q = CAPTION_PREFIX.format(caption=caption or "") + q
    if mode == "agentic" and condition == "vision":
        return q + AGENT_PROTOCOL.format(fmt=inst.answer.format_hint)
    return q + ANSWER_PROTOCOL.format(fmt=inst.answer.format_hint)
