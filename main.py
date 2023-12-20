from langchain.llms.openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
import argparse
from dotenv import load_dotenv
import os

load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY")
IS_DEBUG_MODE = os.getenv("DEBUG") == "true"

parser = argparse.ArgumentParser()
parser.add_argument("--language", default="python")
parser.add_argument("--task", default="Print hello world")
args = parser.parse_args()


llm = OpenAI(
    api_key=API_KEY,
)

code_prompt = PromptTemplate(
    template="Write a program in {language} that will {task}",
    input_variables=["language", "task"],
)

code_chain = LLMChain(llm=llm, prompt=code_prompt, output_key="code")

test_prompt = PromptTemplate(
    template="Write a test for {task} in {language}",
    input_variables=["task", "language"],
)


test_chain = LLMChain(
    llm=llm, prompt=test_prompt, output_key="test", verbose=IS_DEBUG_MODE
)


chain = SequentialChain(
    chains=[code_chain, test_chain],
    input_variables=["language", "task"],
    output_variables=["code", "test"],
)

result = chain({"language": args.language, "task": args.task})
print("Task:\n", result["code"])
print("Test:\n", result["test"])
