import re
from typing import Tuple, List

class Guardrails:
    # Списки запрещённых тем и слов
    forbidden_topics: List[str] = [
        "насилие", "терроризм", "наркотики", "самоубийство",
        "жестокость", "оружие", "порнография"
    ]
    forbidden_words: List[str] = [
        "идиот", "тупой", "убей", "дурак", "дебил"
    ]

    @classmethod
    def validate_input(cls, text: str) -> Tuple[bool, str]:
        """Проверяет входящий запрос. Возвращает (true, "") если всё хорошо."""
        lower_text = text.lower()
        for topic in cls.forbidden_topics:
            if topic in lower_text:
                return False, f"В запросе обнаружена запрещённая тема: '{topic}'. Пожалуйста, переформулируйте."
        return True, ""

    @classmethod
    def validate_output(cls, text: str) -> Tuple[bool, str]:
        """Проверяет сгенерированный ответ на наличие нежелательных слов."""
        for word in cls.forbidden_words:
            if re.search(rf'\b{word}\b', text, re.IGNORECASE):
                return False, f"В ответе обнаружено недопустимое слово: '{word}'."
        return True, ""