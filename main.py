from langchain.chat_models import ChatOpenAI
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder
)
from datetime import date
from langchain.schema import SystemMessage
from langchain.agents import OpenAIFunctionsAgent, AgentExecutor
from dotenv import load_dotenv
from tools.milvus import run_filter_tool, search_milvus_without_filter_tool

load_dotenv()

chat = ChatOpenAI()
prompt = ChatPromptTemplate(
    messages=[
        SystemMessage(content=(
            f"""
            You are an AI that can request more data from a Milvus Database. 
            As much as possible, you should assume that user's queries are mostly in context of Mastercard.
            Users will ask you questions related to Mastercard's initiative, projects, among other generic mastercard related questions.
            The Milvus database can give you the information you need to answer such questions.
            Keep is Mind that today's date is {date.today()}.
            Your job is to use a tool (amongst all available) by providing it **SYNTACTICALLY VALID** boolean Milvus filter expression.
            The tools will use the filter you give them to filterout irrelevant information and provide you with most relevant 
            context, that you can then use to answer the user queries.
            Keep in mind that the Milvus db schema has only these two columns - year and country. 
            So the filter expression you give back to the tools, should - 
            a) Be a **SYNTACTICALLY VALID** boolean milvus filter expression only.
            b) Only use be based on the above mentioned two columns.
            
            You'll find detailed examples of the filter expressions below - 
            [Example #1]
            User: What were the key initiatives by mastercard in past 5 years?
            Your response: "year >= '2019' && year <= '2024'"

            [Example #2]
            User: Summarize main projects of mastercard in past three years.
            Your response: "year >= '2021' && year <= '2024'"

            [Example #3]
            User: What were mastercards main projects in Ghana?
            Your response: "country in ['Ghana']"

            [Example #4]
            User: Give me all projects/initiatives Mastercard did in Uganda in last two years.
            Your response: "year >= '2022' && country in ['Uganda']"

            [Example #5]
            User: What were mastercard's main programs in 2018 for Uganda?
            Your response: "year == '2018' && country in ['Uganda']"

            [Example #6]
            User: List out some main events that happend in 2018 or uganda.
            Your response: "year == '2018' || country in ['uganda']

            [Example #7]
            User: key initiatives in ghana or uganda.
            Your response: "country in ['ghana'] || country in ['uganda']

            [Example #8]
            User: main projects in ghana and uganda after 2020
            Your response: "year >= '2020' && country in ['ghana', 'uganda']
            """
            )),
        HumanMessagePromptTemplate.from_template("{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ]
)

tools = [run_filter_tool, search_milvus_without_filter_tool]

agent = OpenAIFunctionsAgent(
    llm = chat,
    prompt=prompt,
    tools=tools
)

agent_executor = AgentExecutor(
    agent=agent,
    verbose=True,
    tools=tools
)

agent_executor("what projects have we worked on in uganda the past 5 years?")