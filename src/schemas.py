from __future__ import annotations

"""
Схемы данных для интеграции LLM-модуля с agile-доской.

Что важно интегратору:
- этот файл задает контракт входа и выхода LLM-модуля;
- prompt в llm.py должен строиться на основе этих схем, а не на хардкоде;
- если меняются схемы здесь, автоматически должен меняться и ожидаемый формат
  данных для модели.

Общая логика:
- Task / LLMAddTask / LLMChangeTask описывают задачи;
- LLMContext описывает контекст, который передается модели;
- ChatRequest и EncouragementRequest — входные запросы в routes;
- LLMResponse — единственный допустимый формат ответа модели.
"""

from datetime import date, datetime, UTC
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field


class TaskStatus(str, Enum):
    """Допустимые статусы задачи."""

    TODO = "TODO"
    IN_PROGRESS = "In Progress"
    FINISHED = "Finished"


class TaskBase(BaseModel):
    """
    Общая схема задачи без системного id.

    Используется как базовый набор полей для:
    - новых задач;
    - изменяемых задач;
    - задач в контексте.
    """

    title: str = Field(..., description="Название задачи.")
    description: str = Field(..., description="Описание задачи.")
    parent_id: Optional[int] = Field(
        default=None,
        description="ID родительской задачи. Для корневой задачи — null.",
    )
    start_date: date = Field(..., description="Дата начала задачи.")
    end_date: date = Field(..., description="Дата окончания задачи.")
    status: TaskStatus = Field(..., description="Статус задачи.")


class Task(TaskBase):
    """
    Существующая задача в системе.

    Используется в контексте модели и во внутренней логике модуля.
    """

    id: int = Field(..., description="ID задачи.")


class LLMAddTask(TaskBase):
    """
    Задача, которую модель предлагает создать.

    Используется в LLMResponse.add_tasks.
    """

    pass


class LLMChangeTask(TaskBase):
    """
    Задача, которую модель предлагает изменить.

    Используется в LLMResponse.change_tasks.
    """

    id: int = Field(..., description="ID существующей задачи.")


class ChatMessage(BaseModel):
    """
    Одно сообщение из истории чата.

    Используется в LLMContext.chat_history.
    """

    role: str = Field(..., description="Роль сообщения: user / assistant / system.")
    content: str = Field(..., description="Текст сообщения.")


class ProjectContext(BaseModel):
    """
    Общая информация о проекте.

    Вынесена отдельно, чтобы не дублировать project_name в каждой задаче.
    """

    project_id: int = Field(..., description="ID проекта.")
    project_name: str = Field(..., description="Название проекта.")


class LLMContext(BaseModel):
    """
    Контекст, который передается модели.

    Содержит:
    - информацию о проекте;
    - последние сообщения чата;
    - текущий список задач проекта;
    - текущую дату.
    """

    project: ProjectContext = Field(..., description="Информация о текущем проекте.")
    chat_history: List[ChatMessage] = Field(
        default_factory=list,
        description="Последние сообщения чата.",
        max_length=5,
    )
    project_tasks: List[Task] = Field(
        default_factory=list,
        description="Список задач текущего проекта.",
    )
    current_date: date = Field(..., description="Текущая дата.")


class ChatRequest(BaseModel):
    """
    Запрос в chat endpoint.

    Используется для обработки нового сообщения пользователя.
    """

    user_message: str = Field(..., description="Новое сообщение пользователя.")
    context: LLMContext = Field(..., description="Контекст для модели.")


class EncouragementRequest(BaseModel):
    """
    Запрос в encouragement endpoint.

    Используется для генерации автоматического подбадривающего сообщения.
    """

    context: LLMContext = Field(..., description="Контекст для модели.")


class LLMResponse(BaseModel):
    """
    Строгий формат ответа модели.

    Только эти поля допустимы в ответе LLM.
    """

    model_config = ConfigDict(extra="forbid")

    message_to_user: Optional[str] = Field(
        default=None,
        description="Ответ пользователю.",
    )
    add_tasks: List[LLMAddTask] = Field(
        default_factory=list,
        description="Задачи, которые нужно создать.",
    )
    change_tasks: List[LLMChangeTask] = Field(
        default_factory=list,
        description="Задачи, которые нужно изменить.",
    )
    delete_tasks: List[int] = Field(
        default_factory=list,
        description="ID задач для удаления. Сначала дочерние, потом родительские.",
    )


class LLMEnvelope(BaseModel):
    """
    Обертка над ответом модели.

    Нужна только если требуется хранить метаданные вызова.
    Если не нужна — можно не использовать.
    """

    model_name: Optional[str] = Field(
        default=None,
        description="Имя модели.",
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Время создания ответа в UTC.",
    )
    response: LLMResponse = Field(..., description="Ответ модели.")


def get_llm_response_json_schema() -> dict:
    """Вернуть JSON Schema для LLMResponse."""
    return LLMResponse.model_json_schema()


def get_chat_request_json_schema() -> dict:
    """Вернуть JSON Schema для ChatRequest."""
    return ChatRequest.model_json_schema()


def get_encouragement_request_json_schema() -> dict:
    """Вернуть JSON Schema для EncouragementRequest."""
    return EncouragementRequest.model_json_schema()