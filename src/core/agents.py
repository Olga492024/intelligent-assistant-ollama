from langchain_ollama import ChatOllama
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.tools import Tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from .config import ollama_model, ollama_temperature, ollama_top_p

def create_researcher_agent(retriever_tool):
    """Создаёт агента-исследователя, использующего rag и cot."""
    llm = ChatOllama(
        model=ollama_model,
        temperature=ollama_temperature,
        top_p=ollama_top_p
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", """Ты - исследователь-аналитик. Твоя задача - дать точный и обоснованный ответ на русском языке.
Используй инструмент поиска по базе знаний для получения фактов.
Важно: Рассуждай пошагово (chain-of-thought). Сначала проанализируй вопрос, потом извлеки релевантную информацию, затем сформулируй ответ, опираясь только на найденные данные.
Если информации недостаточно, честно скажи об этом. Не выдумывай факты.
Отвечай строго на русском языке."""),
        MessagesPlaceholder(variable_name="chat_history", optional=True),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ])

    agent = create_openai_tools_agent(llm, [retriever_tool], prompt)
    return AgentExecutor(agent=agent, tools=[retriever_tool], verbose=True, handle_parsing_errors=True)

def create_factchecker_agent():
    """Создаёт цепочку для проверки фактов."""
    llm = ChatOllama(
        model=ollama_model,
        temperature=0.0,  # минимальная креативность для проверки
        top_p=ollama_top_p
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", """Ты - строгий фактчекер. Тебе будет передан черновой ответ исследователя и контекст, из которого он был получен.
Проверь, соответствует ли ответ информации из контекста. Укажи на любые несоответствия, галлюцинации или выдуманные факты.
В конце дай вердикт одним из двух слов: 'подтверждён' или 'содержит галлюцинации', после чего кратко объясни причину.
Отвечай строго на русском языке."""),
        ("human", "Черновой ответ: {draft}\n\nКонтекст: {context}")
    ])

    return prompt | llm