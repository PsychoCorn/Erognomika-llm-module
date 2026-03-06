from __future__ import annotations

import json
from typing import Any

from openai import OpenAI
from pydantic import ValidationError

try:
    from config import Settings
    from schemas import ChatRequest, EncouragementRequest, LLMResponse
except ImportError:  # pragma: no cover
    from src.config import Settings
    from src.schemas import ChatRequest, EncouragementRequest, LLMResponse


def _json_dumps(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True)


def build_prompt(request: ChatRequest) -> list[dict[str, str]]:
    response_schema = LLMResponse.model_json_schema()
    request_payload = request.model_dump(mode="json")

    system_prompt = (
        "Ты — LLM-модуль внутри agile-доски задач.\n"
        "Твоя задача — анализировать сообщение пользователя и контекст проекта, "
        "после чего возвращать строго валидный JSON-объект.\n\n"
        "Обязательные правила:\n"
        "1. Верни только JSON без markdown, без пояснений, без префиксов, без ```.\n"
        "2. JSON должен соответствовать указанной ниже JSON Schema.\n"
        "3. Не добавляй поля, которых нет в схеме.\n"
        "4. Если какое-то действие не требуется, используй пустой список или null "
        "в соответствии со схемой.\n"
        "5. Поле message_to_user, если оно заполняется, должно быть написано на том же языке, "
        "на котором пользователь написал свое текущее сообщение.\n"
        "6. Если пользователь пишет по-русски — отвечай по-русски. "
        "Если по-английски — отвечай по-английски. "
        "Если язык смешанный, ориентируйся на доминирующий язык текущего сообщения пользователя.\n"
        "7. Не выдумывай несуществующие задачи и изменения без достаточных оснований в сообщении "
        "пользователя и контексте.\n"
        "8. При удалении задач в delete_tasks соблюдай порядок: сначала дочерние задачи, "
        "потом родительские.\n"
        "9. Если пользователь просто общается и не просит менять задачи, "
        "можно вернуть только message_to_user и пустые списки действий.\n"
        "10. Учитывай текущую дату из контекста при формировании сроков и ответов.\n\n"
        "JSON Schema ожидаемого ответа:\n"
        f"{_json_dumps(response_schema)}"
    )

    user_prompt = (
        "Ниже передан входной запрос для обработки.\n"
        "Проанализируй его и верни только JSON, соответствующий схеме из system message.\n\n"
        "Структура входных данных:\n"
        f"{_json_dumps(request_payload)}"
    )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def build_encouragement_prompt(request: EncouragementRequest) -> list[dict[str, str]]:
    response_schema = LLMResponse.model_json_schema()
    request_payload = request.model_dump(mode="json")

    system_prompt = (
        "Ты — LLM-модуль внутри agile-доски задач.\n"
        "Сейчас твоя задача — автоматически сгенерировать короткое, уместное, "
        "дружелюбное и подбадривающее сообщение для пользователя на основе контекста проекта.\n\n"
        "Обязательные правила:\n"
        "1. Верни только JSON без markdown, без пояснений, без префиксов, без ```.\n"
        "2. JSON должен соответствовать указанной ниже JSON Schema.\n"
        "3. Не добавляй поля, которых нет в схеме.\n"
        "4. Если не требуется изменение задач, верни пустые списки add_tasks, change_tasks, delete_tasks.\n"
        "5. Поле message_to_user должно быть написано на языке пользователя.\n"
        "6. Язык определяй по истории чата. Если язык смешанный, ориентируйся на доминирующий язык последних сообщений.\n"
        "7. Сообщение должно быть естественным, не слишком длинным, без канцелярита.\n"
        "8. Не упоминай, что ты модель, ИИ, ассистент или LLM.\n"
        "9. Не придумывай факты, которых нет в контексте.\n"
        "10. Не используй токсичный, пассивно-агрессивный или слишком фамильярный тон.\n"
        "11. Если в контексте мало информации, все равно верни нейтральное короткое подбадривающее сообщение.\n"
        "12. Учитывай текущую дату из контекста, если это помогает сделать сообщение уместнее.\n\n"
        "JSON Schema ожидаемого ответа:\n"
        f"{_json_dumps(response_schema)}"
    )

    user_prompt = (
        "Ниже передан контекст для генерации автоматического подбадривающего сообщения.\n"
        "Проанализируй его и верни только JSON, соответствующий схеме из system message.\n\n"
        "Структура входных данных:\n"
        f"{_json_dumps(request_payload)}"
    )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def _call_model(
    messages: list[dict[str, str]],
    *,
    model: str,
    api_key: str,
    base_url: str,
    temperature: float,
    max_tokens: int,
    timeout: float,
) -> str:
    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
        timeout=timeout,
    )

    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        response_format={"type": "json_object"},
    )

    if not completion.choices:
        raise ValueError("Модель не вернула ни одного choice.")

    content = completion.choices[0].message.content
    if content is None or not str(content).strip():
        raise ValueError("Модель вернула пустой content.")

    return content


def parse_llm_response(raw_text: str) -> LLMResponse:
    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"LLM вернула невалидный JSON: {exc}") from exc

    try:
        return LLMResponse.model_validate(payload)
    except ValidationError as exc:
        raise ValueError(
            f"LLM вернула JSON, не соответствующий схеме: {exc}"
        ) from exc


def generate_chat_response(
    request: ChatRequest,
    settings: Settings,
) -> LLMResponse:
    messages = build_prompt(request)
    raw_text = _call_model(
        messages,
        model=settings.llm_model,
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
        timeout=settings.llm_timeout,
    )
    return parse_llm_response(raw_text)


def generate_encouragement_response(
    request: EncouragementRequest,
    settings: Settings,
) -> LLMResponse:
    messages = build_encouragement_prompt(request)
    raw_text = _call_model(
        messages,
        model=settings.llm_model,
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
        timeout=settings.llm_timeout,
    )
    return parse_llm_response(raw_text)