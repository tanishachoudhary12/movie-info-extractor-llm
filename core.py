from dotenv import load_dotenv
load_dotenv()
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel
from typing import List, Optional
from langchain_core.output_parsers import PydanticOutputParser

class Movie(BaseModel):
    title: str
    release_year: Optional[int]
    genre: List[str]
    director: Optional[str]
    cast: List[str]
    rating: Optional[float]
    summary: Optional[str] 
     
parser = PydanticOutputParser(pydantic_object=Movie)

model = ChatGroq(model="llama-3.1-8b-instant")

prompt = ChatPromptTemplate.from_messages([
    ('system', """
     Extraxt movie infrormation from the paragraph 
     {format_instructions}
     """),
    ('human', "{paragraph}")
])


para = input("Give your paragraph")
final_prompt = prompt.invoke(
    {"paragraph" : para, 
     "format_instructions" : parser.get_format_instructions()}
)
res = model.invoke(final_prompt)
print(res.content)