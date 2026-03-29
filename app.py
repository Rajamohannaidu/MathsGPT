## ============================================================
## Text To Math Problem Solver & Data Search Assistant
## Fixed for LangChain v1 — all deprecated APIs updated
## ============================================================

import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate              # Fix 1: langchain.prompts → langchain_core.prompts
from langchain_core.output_parsers import StrOutputParser      # Fix 2: replaces LLMChain
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_classic.chains import LLMMathChain              # Fix 3: LLMMathChain → langchain_classic
from langchain_classic.agents import initialize_agent, AgentType  # Fix 4: moved to langchain_classic
from langchain_classic.callbacks import StreamlitCallbackHandler  # Fix 5: moved to langchain_classic
from langchain_core.tools import Tool

# ── Streamlit config ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Text To Math Problem Solver And Data Search Assistant",
    page_icon="🧮"
)
st.title("🧮 Text To Math Problem Solver Using Llama 3")

groq_api_key = st.sidebar.text_input(label="Groq API Key", type="password")

if not groq_api_key:
    st.info("Please add your Groq API key to continue.")
    st.stop()

# Fix 6: Gemma2-9b-It decommissioned → llama-3.3-70b-versatile
# (70b recommended for math reasoning accuracy)
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    groq_api_key=groq_api_key,
    streaming=False   # Fix 7: streaming=False avoids Groq APIError with tool calls
)

# ── Wikipedia tool ────────────────────────────────────────────────────────────
wikipedia_wrapper = WikipediaAPIWrapper()
wikipedia_tool = Tool(
    name="Wikipedia",
    func=wikipedia_wrapper.run,
    description="A tool for searching the internet to find various information on topics mentioned."
)

# ── Calculator tool ───────────────────────────────────────────────────────────
math_chain = LLMMathChain.from_llm(llm=llm)
calculator = Tool(
    name="Calculator",
    func=math_chain.run,
    description="A tool for answering math related questions. Only input mathematical expressions need to be provided."
)

# ── Reasoning tool ────────────────────────────────────────────────────────────
# Fix 8: LLMChain replaced with LCEL prompt | llm | StrOutputParser()
reasoning_prompt = PromptTemplate(
    input_variables=["question"],
    template="""
    You are an agent tasked for solving users mathematical questions.
    Logically arrive at the solution and provide a detailed explanation
    and display it point wise for the question below.
    Question: {question}
    Answer:
    """
)
reasoning_chain = reasoning_prompt | llm | StrOutputParser()

reasoning_tool = Tool(
    name="Reasoning Tool",
    func=lambda q: reasoning_chain.invoke({"question": q}),
    description="A tool for answering logic-based and reasoning questions."
)

# ── Agent ─────────────────────────────────────────────────────────────────────
assistant_agent = initialize_agent(
    tools=[wikipedia_tool, calculator, reasoning_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=False,
    handle_parsing_errors=True
)

# ── Chat history ──────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hi! I'm a Math chatbot who can answer all your math questions."}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# ── Input ─────────────────────────────────────────────────────────────────────
question = st.text_area(
    "Enter your question:",
    "I have 5 bananas and 7 grapes. I eat 2 bananas and give away 3 grapes. "
    "Then I buy a dozen apples and 2 packs of blueberries. "
    "Each pack of blueberries contains 25 berries. "
    "How many total pieces of fruit do I have at the end?"
)

if st.button("Find My Answer"):
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Generating response..."):
            st.session_state.messages.append({"role": "user", "content": question})
            st.chat_message("user").write(question)

            st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)

            # Fix 9: Pass only the current question, not full message history
            # assistant_agent.run(st.session_state.messages) was wrong —
            # ZERO_SHOT_REACT_DESCRIPTION expects a plain string input
            response = assistant_agent.run(question, callbacks=[st_cb])

            st.session_state.messages.append({"role": "assistant", "content": response})
            st.write("### Response:")
            st.success(response)