export interface AgentPrompts {
  [key: string]: { [key: string]: string };
}

export interface PersonaConfig {
  role: string;
  icon: string;
  bias_instruction: string;
}

export interface LLMConfig {
  LLM_MAX_RETRIES: number;
  OPENAI_MODEL: string;
  TIMEOUT: number | null;
  CACHE: boolean;
  TEMPERATURE: number;
  MAX_COMPLETION_TOKENS: number | null;
}

export interface GraphConfig {
  ACTION_GRAPH_MAX_RETRIES: number;
  TASK_GRAPH_MAX_RETRIES: number;
}

export interface AppSettings {
  prompts: AgentPrompts;
  personas: PersonaConfig[];
  llm_config: LLMConfig;
  graph_config: GraphConfig;
}

export const DEFAULT_SETTINGS: AppSettings = {
  prompts: {
    analysis: { system_prompt: "", prompt_user_instruction: "" },
    code: { system_prompt: "", system_prompt_replan: "" },
    master: { system_prompt: "" },
  },
  personas: [],
  llm_config: {
    LLM_MAX_RETRIES: 2,
    OPENAI_MODEL: "gpt-4o-mini",
    TIMEOUT: 300,
    CACHE: false,
    TEMPERATURE: 0.3,
    MAX_COMPLETION_TOKENS: null,
  },
  graph_config: {
    ACTION_GRAPH_MAX_RETRIES: 5,
    TASK_GRAPH_MAX_RETRIES: 3,
  },
};
