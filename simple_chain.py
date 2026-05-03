from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

input_data = """The Man Who Wanted to Change Everything

Elon Musk was born on June 28, 1971, in Pretoria, South Africa. Even as a child, he was obsessed with books and technology — at just 12 years old, he wrote his first video game and sold it for $500. At 17, he moved to Canada, and then to the United States, chasing his dream.

The first money came fast. In 1995, he founded Zip2 and sold it for $22 million, then created X.com — the future PayPal. When eBay acquired PayPal in 2002 for $1.5 billion, Musk took his share and immediately put almost everything back on the line.
​

He founded SpaceX with a dream of reaching Mars. The first three rockets exploded. The company was on the edge of collapse. The fourth launch in 2008 saved everything — it became the first privately built rocket to reach orbit in history.
​

At the same time, he invested in Tesla, which nearly died during the 2008 financial crisis. Musk sold personal assets just to keep the company alive. Tesla survived — and became a symbol of the electric revolution.
​

Today he is the wealthiest person on the planet, owns the social network X (formerly Twitter), develops brain-computer interfaces through Neuralink, and builds tunnels via The Boring Company. A man who simply does not know how to do anything halfway."""

if __name__ == "__main__":
    summary_template = """
        given the information {information} about a person from I want you to create:
        1. A short summary.
        2. two interesting facts about them
        """

    summery_prompt_template = PromptTemplate(input_variables=["information"], template=summary_template)
    llm = ChatOpenAI(temperature=0, model_name='gpt-5.4-mini')
    chain = summery_prompt_template | llm | StrOutputParser()
    print(chain.invoke(input={"information": input_data}))
