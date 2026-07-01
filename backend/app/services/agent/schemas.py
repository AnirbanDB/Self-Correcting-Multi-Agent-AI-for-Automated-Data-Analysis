from enum import Enum
from typing import Optional, TypedDict, List, Any
from pydantic import BaseModel, Field

# --- Enums ---

class TaskStatus(str, Enum):
    """Enum for task status values."""
    SUCCESS = "success"
    FAILED = "failed" 
    PENDING = "pending"
    RUNNING = "running"

class TaskType(str, Enum):
    """Enum for task type values."""
    DATA_LOADING = "data_loading"
    EXPLORATION = "exploration"
    FEATURE_ENGINEERING = "feature_engineering"
    MODEL_TRAINING = "model_training"
    EVALUATION = "evaluation"
    VISUALIZATION = "visualization"

# --- Pydantic Schemas ---

class PydanticActionNode(BaseModel):
    action_id: int = Field(..., description="The sequential ID of the action node in numbers only.")
    description: str = Field(..., description="A brief, natural language that describes what the action node does.")
    code: str = Field(..., description="A valid, executable python code snippet")

class PydanticActionGraph(BaseModel):
    task_nodes: List[PydanticActionNode] = Field(default=[] ,  description="List of action nodes for the specific task")

class PydanticTaskNode(BaseModel):
    task_id: str = Field(..., description="Unique id for the task in numbers. Examples: 1, 2, 100")
    task_name: str = Field(..., description="Unique name for the task.")
    dependencies: List[str] = Field(..., description="A list of unique ids...")
    instruction: str = Field(..., description="A concise instruction on what the task should achieve")
    task_type: TaskType = Field(description="Primary task type of the task")
    output: str = Field(..., description="description of what data is expected")

class PydanticTaskGraph(BaseModel):
    task_nodes: List[PydanticTaskNode] = Field(..., description="A list of task nodes the task should be breaken down into.")

class PydanticEditAction(str, Enum):
    ADD = "add"       
    MODIFY = "modify" 
    DELETE = "delete" 

class PydanticGraphEdit(BaseModel):
    action: PydanticEditAction = Field(..., description="The type of change to apply.")
    # For DELETE, only need the ID.
    # For ADD/MODIFY, need the full node details.
    task: Optional[PydanticTaskNode] = Field(None, description="The task details. Required for ADD and MODIFY.")
    target_task_id: Optional[str] = Field(None, description="The specific ID of the task to DELETE.")

class PydanticGraphModificationPlan(BaseModel):
    reasoning: str = Field(..., description="Brief explanation of why these changes meet the new requirement.")
    edits: List[PydanticGraphEdit] = Field(..., description="List of atomic edits to apply to the graph.")

class PydanticMasterResult(BaseModel):
    response: str = Field(..., description="analysis result summary...")

class PydanticDiagramResult(BaseModel):
    filename: str = Field(..., description="The filename of the figure.")
    text: str = Field(..., description="The explanation of the diagram.")

class PydanticAnalysisResult(BaseModel):
    summary: str = Field(..., description="The type of change to apply.")
    # figures: List[PydanticDiagramResult] = Field(..., description="List of handpicked diagrams that is crucial to help with explanation.")

class PydanticForecastResult(BaseModel):
    forecast: List[str] = Field(..., description="The list of forecast result.")
    text: str = Field(..., description="The explanation of the choices.")

# --- TypedDicts ---

class AgentMessage(TypedDict):
    """Represents a structured message from an agent to the shared state."""
    sender: str
    content: Any

class PersonaResponse(TypedDict):
    role: str
    persona: str
    icon: str
    content: PydanticAnalysisResult | PydanticForecastResult

# --- Bias Evaluation ---
class BiasMetrics(BaseModel):
    bias_score: float       # -1 (Bear) to +1 (Bull)
    neutrality_index: float # 0 to 1 (Success metric)
    polarity: float         # Raw sentiment

class AgentEvaluationResult(BaseModel):
    role: str
    persona: str
    icon: str
    content: str
    metrics: BiasMetrics

# --- Master Agent ---
class GlobalAgentState(TypedDict):
    """A comprehensive state for a data science pipeline."""
    sess_id: str
    run_id: str
    requirement: str
    num_steps: int
    raw_data_filenames: List[str]
    visualization_paths: List[str]
    analysis_result: List[AgentEvaluationResult]
    neutral_report: Optional[str]
    agent_messages: List[AgentMessage]

class MultiPersonaResponse(TypedDict):
    run_id: str
    text: str
    perspectives: List[AgentEvaluationResult]
    figures: List[str] = []