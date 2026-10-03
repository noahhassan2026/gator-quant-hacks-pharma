import os
import time
from google import genai
from google.genai import types
from google.genai import errors
from pydantic import BaseModel, Field

class LegalAlphaSignal(BaseModel):
    ticker: str = Field(description="Associated stock ticker for the branded pharma company")
    case_name: str = Field(description="Name of the patent litigation matter")
    judge_name: str = Field(description="Presiding judge")
    milestone_type: str = Field(description="e.g., Markman Ruling, Motion for Summary Judgment, Case Assignment")
    alpha_score: float = Field(
        description="Float between -1.0 (strongly favors generic/patent invalid) to +1.0 (strongly favors branded innovator/patent valid)"
    )
    confidence: float = Field(description="Model confidence from 0.0 to 1.0")
    key_legal_finding: str = Field(description="1-2 sentence distillation of the judicial construction or ruling")

def analyze_legal_text(document_text: str, client: genai.Client) -> LegalAlphaSignal:
    system_instruction = (
        "You are an expert pharmaceutical patent litigator and quantitative trading analyst. "
        "Analyze the provided Hatch-Waxman ANDA court document or judge assignment history. "
        "Score the ruling strictly from the perspective of the branded patent owner (innovator). "
        "A positive score (+0.1 to +1.0) means the patent is likely upheld or claims were narrowly construed in the innovator's favor. "
        "A negative score (-0.1 to -1.0) means the patent is vulnerable to invalidation or generic entry is accelerated."
    )

    max_retries = 5
    base_wait_time = 4  # seconds
    
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model="gemini-3.8-flash",
                contents=document_text,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    response_mime_type="application/json",
                    response_schema=LegalAlphaSignal,
                    temperature=0.1,
                ),
            )
            return LegalAlphaSignal.model_validate_json(response.text)
            
        except errors.APIError as e:
            if getattr(e, 'code', None) in [503, 429]:
                wait = base_wait_time * (2 ** attempt) 
                print(f"[Network] API busy (Error {e.code}). Retrying in {wait} seconds... (Attempt {attempt + 1}/{max_retries})")
                time.sleep(wait)
            else:
                raise e
                
    raise Exception("Gemini API is currently overloaded. Pipeline paused.")


# --- MAKE SURE THIS SECTION IS AT THE VERY BOTTOM AND TOUCHES THE LEFT EDGE ---
if __name__ == "__main__":
    sample_ruling = """
    IN THE UNITED STATES DISTRICT COURT FOR THE DISTRICT OF DELAWARE
    BRISTOL-MYERS SQUIBB CO. v. APOTEX INC.
    Civil Action No. 22-cv-01124-CFC
    
    MEMORANDUM ORDER CONCERNING CLAIM CONSTRUCTION:
    The Court has reviewed the disputed terms of U.S. Patent No. 8,461,199 ('199 patent).
    With respect to the term 'crystalline anhydrous salt having an XRPD pattern of peaks at 8.2 and 14.5 degrees',
    the Court rejects Defendant Apotex's proposed broad construction encompassing amorphous hydrates.
    The Court adopts Plaintiff BMS's proposed narrow construction, restricting the patent claim strictly to 
    pure anhydrous polymorph Form I. Consequently, defendant's bioequivalence argument under the doctrine of equivalents is curtailed.
    """
    
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("Missing GEMINI_API_KEY environment variable. Run 'export GEMINI_API_KEY=your_key_here' first.")
        
    client = genai.Client(api_key=api_key)
    
    print("Sending legal filing to Gemini API...")
    signal = analyze_legal_text(sample_ruling, client)
    
    print("\n--- Parsed Legal Signal ---")
    print(f"Ticker:        {signal.ticker}")
    print(f"Judge:         {signal.judge_name}")
    print(f"Alpha Score:   {signal.alpha_score}")
    print(f"Confidence:    {signal.confidence}")
    print(f"Key Finding:   {signal.key_legal_finding}")