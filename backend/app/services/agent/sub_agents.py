import base64, logging, copy
from typing import List, Literal
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field
from core.config import Persona, default_config
from .schemas import GlobalAgentState, PersonaResponse, PydanticAnalysisResult, PydanticForecastResult
from .utils import load_prompt, SessionWorkspace

OPENAI_API_KEY = default_config.OPENAI_API_KEY

logger = logging.getLogger(default_config.SESS_LOG_NAME)

ProviderType = Literal["openai", "gemini", "grok"]

class DiagramSelection(BaseModel):
    selected_paths: List[str] = Field(description="The list of file paths that are most relevant")
    reasoning: str = Field(description="Brief reason for this selection")

class AnalysisAgent:
    """A class to perform analysis based on code outputs."""
    def __init__(self, model:str, provider: ProviderType = "openai"):
        self.system_prompt = load_prompt(agent_name=default_config.AGENT_NAME_ANALYSIS)
        self.system_prompt_forecast = load_prompt(agent_name=default_config.AGENT_NAME_ANALYSIS,key=default_config.PROMPT_KEY_FORECAST)
        self.prompt_user_instruction = load_prompt(agent_name=default_config.AGENT_NAME_ANALYSIS, key=default_config.PROMPT_KEY_ANALYSIS_USER)
        self.provider = provider
        self.model_name = model
        self.llm = self._initialize_llm()

    def _initialize_llm(self):
        """Factory method to return the correct LangChain chat model."""
        if self.provider == "openai":
            return ChatOpenAI(
                model=self.model_name,
                openai_api_key=default_config.OPENAI_API_KEY,
                temperature=default_config.OPENAI_TEMPERATURE,
                max_completion_tokens=default_config.OPENAI_MAX_COMPLETION_TOKENS,
                timeout=default_config.OPENAI_TIMEOUT,
                max_retries=default_config.OPENAI_MAX_RETRIES
            )
            
        elif self.provider == "gemini":
            return ChatGoogleGenerativeAI(
                model=self.model_name,
                google_api_key=default_config.GEMINI_API_KEY, # Ensure this exists in your config
                temperature=default_config.OPENAI_TEMPERATURE,
                max_output_tokens=default_config.OPENAI_MAX_COMPLETION_TOKENS,
                timeout=default_config.OPENAI_TIMEOUT,
                convert_system_message_to_human=True # Sometimes needed for older Gemini models
            )

        elif self.provider == "grok":
            return ChatOpenAI(
                model=self.model_name, # e.g., "grok-vision-beta"
                openai_api_key=default_config.XAI_API_KEY, # Ensure this exists in your config
                base_url="https://api.x.ai/v1",
                temperature=default_config.OPENAI_TEMPERATURE,
                max_completion_tokens=default_config.OPENAI_MAX_COMPLETION_TOKENS,
                timeout=default_config.OPENAI_TIMEOUT
            )
            
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")
    
    def _encode_image(self, image_path: str):
        """Helper function to encode image to base64."""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    def select_key_diagrams(self, state: GlobalAgentState, max_items: int = 5) -> List[str]:
        """
        Filters the list of all generated diagrams down to the most relevant ones
        based on the user's requirement.
        """
        all_paths = state.get('visualization_paths', [])
        
        if len(all_paths) <= max_items:
            return all_paths

        structured_curator = self.llm.with_structured_output(DiagramSelection)
        
        prompt = (
            "You are a Senior Editor. A data analysis workflow produced many diagrams.\n"
            f"User Query: '{state['requirement']}'\n\n"
            "Here are the available diagram files:\n"
            f"{all_paths}\n\n"
            f"Select the top {max_items} diagrams that provide the BEST evidence to answer the query.\n"
        )

        try:
            selection: DiagramSelection = structured_curator.invoke([HumanMessage(content=prompt)])
            
            valid_paths = [p for p in selection.selected_paths if p in all_paths]
            if valid_paths:
                return valid_paths
            
            raise RuntimeError(f"no valid paths returned from llm.")
            
        except Exception as e:
            logger.warning(f"{self.__class__.__name__} - Returned last {max_items} items. Something went wrong when selecting key diagrams: {e}")
            return all_paths[-max_items:]
    
    def analyze_diagrams_with_persona(self, diagram_paths: List[str], persona: Persona, user_prompt: str) -> str:
        """
        Encodes the image and asks the vision model a question about it.
        """
        import os

        image_content = []
        for path in diagram_paths:
            filename = os.path.basename(path)
            image_content.extend([
                {
                    "type": "text", 
                    "text": f"Image Filename: {filename}"
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{self._encode_image(path)}"
                    }
                }
            ])

        dynamic_system_prompt = (
            f"You are acting as: {persona.role}.\n"
            f"Your Core Bias: {persona.injected_persona}.\n"
            f"{self.system_prompt}"
            f"{self.prompt_user_instruction}"
        )

        user_instruction = (
            f"User Request: {user_prompt}\n\n"
        )

        messages = [
            SystemMessage(content=dynamic_system_prompt),
            HumanMessage(content=[
                {"type": "text", "text": user_instruction},
                *image_content 
            ])
        ]

        structured_llm = self.llm.with_structured_output(PydanticAnalysisResult)

        pydantic_response: PydanticAnalysisResult = structured_llm.invoke(messages) # type: ignore

        return pydantic_response
    
    def forecast_with_persona(self, diagram_paths: List[str], persona: Persona, user_prompt: str) -> str:
        """
        Encodes the image and asks the vision model a question about it.
        """
        import os

        image_content = []
        for path in diagram_paths:
            filename = os.path.basename(path)
            image_content.extend([
                {
                    "type": "text", 
                    "text": f"Image Filename: {filename}"
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{self._encode_image(path)}"
                    }
                }
            ])

        dynamic_system_prompt = (
            f"You are acting as: {persona.role}.\n"
            f"Your Core Bias: {persona.injected_persona}.\n"
            f"{self.system_prompt_forecast}"
        )

        user_instruction = (
            f"User Request: {user_prompt}\n\n"
        )

        messages = [
            SystemMessage(content=dynamic_system_prompt),
            HumanMessage(content=[
                {"type": "text", "text": user_instruction},
                *image_content 
            ])
        ]

        structured_llm = self.llm.with_structured_output(PydanticForecastResult)

        pydantic_response: PydanticForecastResult = structured_llm.invoke(messages) # type: ignore

        return pydantic_response
    

    def analyze_diagrams_with_all_personas(self, workflow_state, diagram_paths: list[str], personas: List[Persona], logger) -> List[PersonaResponse]:
        """
        Consults all personas concurrently using multithreading to reduce wait time.
        """
        import concurrent.futures

        collected_insights: List[PersonaResponse] = []
        
        def _consult_persona(persona_config):
            try:
                # Perform the Analysis (Blocking I/O)
                analysis_content = self.analyze_diagrams_with_persona(
                    diagram_paths=diagram_paths, 
                    persona=persona_config, 
                    user_prompt=workflow_state['requirement']
                )
                
                return PersonaResponse(
                    role=persona_config.role, 
                    persona=persona_config.injected_persona, 
                    icon=persona_config.icon, 
                    content=analysis_content
                )
            except Exception as e:
                logger.error(f"Error consulting {persona_config.role}: {e}")
                return None

        # limit max_workers to avoid hitting OpenAI Rate Limits too hard
        max_workers = min(len(personas), 5) 
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(_consult_persona, p) for p in personas]
            
            # Gather results as they complete
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                if result:
                    collected_insights.append(result)

        return collected_insights
    
    def forecast_diagrams_with_all_persona(self, workflow_state, diagram_paths: list[str], personas: List[Persona], logger) -> List[PersonaResponse]:
        import concurrent.futures

        collected_results: List[PersonaResponse] = []
        
        def _consult_persona(persona_config):
            try:
                # Perform the Analysis (Blocking I/O)
                analysis_content = self.forecast_with_persona(
                    diagram_paths=diagram_paths, 
                    persona=persona_config, 
                    user_prompt=workflow_state['requirement']
                )
                
                return PersonaResponse(
                    role=persona_config.role, 
                    persona=persona_config.injected_persona, 
                    icon=persona_config.icon, 
                    content=analysis_content
                )
            except Exception as e:
                logger.error(f"Error consulting {persona_config.role}: {e}")
                return None

        max_workers = min(len(personas), 5) 
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(_consult_persona, p) for p in personas]
            
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                if result:
                    collected_results.append(result)

        return collected_results 