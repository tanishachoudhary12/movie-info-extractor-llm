import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel
from typing import List, Optional
from langchain_core.output_parsers import PydanticOutputParser

# Load environment variables
load_dotenv()

# ----------- Page Config -----------
st.set_page_config(
    page_title="Movie Info Extractor",
    page_icon="🎬",
    layout="centered"
)

# ----------- Custom Styling -----------
st.markdown("""
    <style>
    .main {
        background-color: #0f172a;
        color: white;
    }
    .stTextArea textarea {
        background-color: #1e293b;
        color: white;
        border-radius: 10px;
    }
    .title {
        text-align: center;
        font-size: 40px;
        font-weight: bold;
        color: #f472b6;
    }
    .subtitle {
        text-align: center;
        font-size: 18px;
        color: #cbd5f5;
        margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# ----------- Header -----------
st.markdown('<div class="title">🎬 Movie Info Extractor</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Because I got tired of reading long movie descriptions 😌</div>', unsafe_allow_html=True)

# ----------- Model Schema -----------
class Movie(BaseModel):
    title: str
    release_year: Optional[int]
    genre: List[str]
    director: Optional[str]
    cast: List[str]
    rating: Optional[float]
    summary: Optional[str]

parser = PydanticOutputParser(pydantic_object=Movie)

# ----------- LLM -----------
model = ChatGroq(model="llama-3.1-8b-instant")

# ----------- Prompt -----------
prompt = ChatPromptTemplate.from_messages([
    ('system', """
    Extract movie information from the paragraph.
    {format_instructions}
    """),
    ('human', "{paragraph}")
])

# ----------- Input UI -----------
user_input = st.text_area("Paste your movie paragraph here 👇", height=200)

# ----------- Button -----------
if st.button("✨ Extract Info"):
    if user_input.strip() == "":
        st.warning("Please enter a paragraph first!")
    else:
        with st.spinner("Analyzing... 👀"):
            final_prompt = prompt.invoke({
                "paragraph": user_input,
                "format_instructions": parser.get_format_instructions()
            })

            res = model.invoke(final_prompt)

            try:
                parsed_output = parser.parse(res.content)

                # ----------- Output UI -----------
                st.success("Done! Here's what I found 💅")

                st.subheader(f"🎥 {parsed_output.title}")

                col1, col2 = st.columns(2)

                with col1:
                    st.write(f"📅 **Year:** {parsed_output.release_year}")
                    st.write(f"🎭 **Genre:** {', '.join(parsed_output.genre)}")

                with col2:
                    st.write(f"🎬 **Director:** {parsed_output.director}")
                    st.write(f"⭐ **Rating:** {parsed_output.rating}")

                st.write(f"👥 **Cast:** {', '.join(parsed_output.cast)}")

                st.markdown("### 📝 Summary")
                st.write(parsed_output.summary)

            except Exception as e:
                st.error("Couldn't parse properly 😭")
                st.code(res.content)