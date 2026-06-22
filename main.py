import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser

app = FastAPI(title="Lyra API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TranslateRequest(BaseModel):
    text: str
    target_language: str
    persona: str

hf_token = os.environ.get("HF_TOKEN")
repo_id = "Qwen/Qwen2.5-14B-Instruct" # Hem çok zeki hem de API için mükemmel hızda

# Modeli genel havuza değil, doğrudan kendi özel API odasına yönlendiriyoruz
direct_url = "https://api-inference.huggingface.co/v1/"
# 1. HUGGING FACE'İ OPENAI GİBİ KULLANAN MODERN API BAĞLANTISI
llm = ChatOpenAI(
    model=repo_id,
    api_key=hf_token,
    base_url=direct_url, # Sihrin gerçekleştiği URL
    max_tokens=512,
    temperature=0.1
)

llm_creative = ChatOpenAI(
    model=repo_id,
    api_key=hf_token,
    base_url=direct_url,
    max_tokens=512,
    temperature=0.7
)

# 2. LANGCHAIN ZİNCİRLERİ (Artık <|im_start|> gibi manuel etiketler yok)
class SemanticFrame(BaseModel):
    core_meaning: str = Field(description="The pure propositional content. DO NOT literally translate idioms, metaphors, or slang. Extract factual meaning.")   
    speech_act: str = Field(description="e.g. request, complaint, compliment, apology, statement")
    entities: str = Field(description="Key named entities or referents, or 'none'")
    valence: str = Field(description="Emotional tone of the original")

extract_parser = JsonOutputParser(pydantic_object=SemanticFrame)

# Modern Chat Mimarisi (System ve User mesajları otomatik ayrılır)
extract_prompt = ChatPromptTemplate.from_messages([
    ("system", "You decompose text into a register-neutral semantic frame.\nCRITICAL INSTRUCTION: You MUST output ONLY valid JSON.\n{format_instructions}"),
    ("user", "Text: {text}")
]).partial(format_instructions=extract_parser.get_format_instructions())

json_chain = extract_prompt | llm | extract_parser

render_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a master of sociolinguistics. Reconstruct the following semantic frame into {target_language}.\nPERSONA: {persona}\nSEMANTIC FRAME: Core meaning: {core_meaning}, Speech act: {speech_act}, Emotion: {valence}\nCRITICAL: Do NOT translate literally. Write a completely natural, single utterance. Do not add any explanations."),
    ("user", "Render the text.")
])

render_chain = render_prompt | llm_creative | StrOutputParser()

# 3. ENDPOINT
@app.post("/translate")
async def process_translation(req: TranslateRequest):
    try:
        data = json_chain.invoke({"text": req.text})
        render_conclusion = render_chain.invoke({
            "target_language": req.target_language,
            "persona": req.persona,
            **data
        })
        return {
            "status": "success",
            "final_translation": render_conclusion,
            "extracted_data": {           # ✅ Bu alan eksikti
                "core_meaning": data.get("core_meaning", ""),
                "speech_act": data.get("speech_act", ""),
                "entities": data.get("entities", "none"),
                "valence": data.get("valence", "Neutral"),
            }
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

import httpx

@app.get("/debug")
async def debug():
    try:
        async with httpx.AsyncClient() as client:
            r = await client.get(
                f"https://api-inference.huggingface.co/models/{repo_id}",
                headers={"Authorization": f"Bearer {hf_token}"},
                timeout=10
            )
            return {"status": r.status_code, "body": r.json()}
    except Exception as e:
        return {"error": str(e)}
