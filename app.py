import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser

app = FastAPI(title="Lyra API")

# Sitenin GitHub Pages adresini buraya ekleyebilirsin, şimdilik herkese açık ('*')
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

# 1. MODELİ İNDİRMİYORUZ, API ÜZERİNDEN BAĞLANIYORUZ
hf_token = os.environ.get("HF_TOKEN")
# Qwen'in çok daha zeki olan 72B (Milyar) versiyonunu veya Mistral kullanabiliriz!
repo_id = "Qwen/Qwen2.5-14B-Instruct" 

llm = HuggingFaceEndpoint(
    repo_id=repo_id,
    max_new_tokens=512,
    temperature=0.1,
    huggingfacehub_api_token=hf_token
)

# Daha yaratıcı çeviriler için ikinci bir LLM örneği
llm_creative = HuggingFaceEndpoint(repo_id=repo_id, max_new_tokens=512, temperature=0.7, huggingfacehub_api_token=hf_token)

# 2. LANGCHAIN ZİNCİRLERİ (Senin yazdığın kusursuz promp'lar)
class SemanticFrame(BaseModel):
    core_meaning: str = Field(description="The pure propositional content. DO NOT literally translate idioms, metaphors, or slang. Extract their actual underlying factual meaning in plain, boring English.")   
    speech_act: str = Field(description="e.g. request, complaint, compliment, apology, statement")
    entities: str = Field(description="Key named entities or referents, comma-separated, or 'none'")
    valence: str = Field(description="Emotional tone of the original")

extract_parser = JsonOutputParser(pydantic_object=SemanticFrame)

extract_template = """<|im_start|>system
You decompose text into a register-neutral semantic frame.
CRITICAL INSTRUCTION: You MUST output ONLY valid JSON.
{format_instructions}<|im_end|>
<|im_start|>user
Text: {text}<|im_end|>
<|im_start|>assistant
"""
extract_prompt = PromptTemplate(template=extract_template, input_variables=["text"], partial_variables={"format_instructions": extract_parser.get_format_instructions()})
json_chain = extract_prompt | llm | extract_parser

render_template = """<|im_start|>system
You are a master of sociolinguistics. Reconstruct the following semantic frame into {target_language}.
SEMANTIC FRAME: Core meaning: {core_meaning}, Speech act: {speech_act}, Emotion: {valence}
PERSONA: {persona}
CRITICAL INSTRUCTION: Do NOT translate literally. Write a completely natural, single utterance. Do not add any explanations.<|im_end|>
<|im_start|>user
Render the text.<|im_end|>
<|im_start|>assistant
"""
render_prompt = PromptTemplate(template=render_template, input_variables=["target_language", "core_meaning", "speech_act", "valence", "persona"])
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
        return {"status": "success", "final_translation": render_conclusion}
    except Exception as e:
        return {"status": "error", "message": str(e)}
