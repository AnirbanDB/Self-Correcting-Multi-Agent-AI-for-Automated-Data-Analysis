import logging
from typing import Any, Dict, List, Union, Literal
from contextlib import contextmanager
from app.services.agent.bias_evaluator import BiasEvaluator
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage, messages_to_dict, messages_from_dict
from core.config import default_config, AppSettings, LlmConfig, GraphConfig, Persona
from .schemas import AgentEvaluationResult, GlobalAgentState, MultiPersonaResponse, PersonaResponse, PydanticAnalysisResult, PydanticDiagramResult
from .utils import load_prompt, SessionWorkspace
from .file_utils import FileUtils
from .graph import TaskGraph
from .sub_agents import AnalysisAgent

c_logger = logging.getLogger(default_config.CENTRAL_LOG_NAME)

class MasterAgent:
    def __init__(self, config: AppSettings, tools=[]):
        """Initialize master agent with LLM client, tools, prompts, and sub-agents."""
        self.graph_config: GraphConfig = config.graph_config
        self.personas: List[Persona] = config.personas
        self.llm_config: LlmConfig = config.llm_config
        self.llm = ChatOpenAI(
            model=self.llm_config.OPENAI_MODEL,
            openai_api_key=self.llm_config.OPENAI_API_KEY,
            temperature=self.llm_config.TEMPERATURE,
            max_completion_tokens=self.llm_config.MAX_COMPLETION_TOKENS,
            timeout=self.llm_config.TIMEOUT,
            max_retries=self.llm_config.LLM_MAX_RETRIES
          )
        self.tools = tools
        self.instructions_ans = load_prompt(agent_name=default_config.AGENT_NAME_MASTER, key=default_config.PROMPT_KEY_MASTER_ANS)
        self.instructions_user_req = load_prompt(agent_name=default_config.AGENT_NAME_MASTER, key=default_config.PROMPT_KEY_MASTER_REQ)
        self.task_graph: TaskGraph = TaskGraph(llm_config=self.llm_config, graph_config=self.graph_config)
        self.analysis_agent = AnalysisAgent(model=default_config.OPENAI_MODEL)
        self.bias_evaluator = BiasEvaluator()
        self.conversation_history: List[BaseMessage] = []
  
    def _get_history_file_path(self, workspace: SessionWorkspace) -> str:
        """
        Constructs the path for the session-level history file.
        It should be outside the specific run folder, in the session root.
        """
        import os
        session_dir = os.path.dirname(workspace.run_base) 
        return os.path.join(session_dir, default_config.FILENAME_SESSION_HISTORY)

    def _load_conversation_history(self, workspace: SessionWorkspace, logger: logging.Logger):
        """Loads existing conversation history from JSON."""
        import os, json
        history_path = self._get_history_file_path(workspace)
        if os.path.exists(history_path):
            try:
                with open(history_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.conversation_history = messages_from_dict(data)
                c_logger.info(f"Loaded {len(self.conversation_history)} messages.")
            except Exception as e:
                c_logger.error(f"Failed to load history: {e}")
                self.conversation_history = []
        else:
            self.conversation_history = []

    def _save_conversation_history(self, workspace: SessionWorkspace, logger: logging.Logger) -> str:
        """Saves current conversation history to JSON."""
        import json
        history_path = self._get_history_file_path(workspace)
        try:
            messages_data = messages_to_dict(self.conversation_history)
            
            with open(history_path, 'w', encoding='utf-8') as f:
                json.dump(messages_data, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved history to {history_path}")
        except Exception as e:
            logger.error(f"Failed to save history: {e}")

    @contextmanager
    def _session_logger(self, workspace: SessionWorkspace):
        """Context manager to handle session-specific logging setup/teardown."""
        log_path = workspace.get_log_path()

        handler = logging.FileHandler(log_path)
        sess_formatter = logging.Formatter(default_config.SESS_LOG_FORMAT)
        handler.setFormatter(sess_formatter)
        handler.setLevel(logging.DEBUG)
        
        sess_logger = logging.getLogger(default_config.SESS_LOG_NAME)
        sess_logger.setLevel(logging.DEBUG)
        sess_logger.addHandler(handler)

        # setup handler
        sess_logger = logging.getLogger(default_config.SESS_LOG_NAME)
        sess_logger.addHandler(handler)
        try:
            yield sess_logger
        finally:
            sess_logger.removeHandler(handler)
            handler.close()

    def _initialize_agent_state(self, workplace:SessionWorkspace, requirement:str, file_list: List[str]) -> GlobalAgentState:
        """Create the initial agent state payload for a run."""
        state = GlobalAgentState(sess_id=workplace.sess_id, run_id=workplace.run_id, requirement=requirement, num_steps=0, raw_data_filenames=file_list, evaluation_results=[], visualization_paths=[], agent_messages=[], neutral_report='this is default report placeholder')
        return state
    
    def _synthesize_reports(self, requirement: str, perspectives: List[PersonaResponse]) -> str:
        """The Neutral Arbitrator Logic"""
    
        # Combine all biased reports into one context block
        debate_transcript = ""
        for p in perspectives:
            debate_transcript += f"--- {p['role']} says: ---\n{p['content']}\n\n"

        system_prompt = (
            "You are the Neutral Arbitrator.\n"
            "You have received conflicting analyses from biased agents.\n"
            "Your Goal:\n"
            "1. Strip away the emotional/subjective language.\n"
            "2. Merge the factual findings into a perfectly neutral executive summary.\n"
            "3. Explicitly mention where the agents disagreed."
            "4. When discussing a specific insight that is visualized in a chart or diagram, insert the chart/diagram immediately after the explanation using the format <<<filename>>>. Do not invent filenames, only use the one listed. Example Output: Here is the summary of Q1.\n <<<sales_q1.png>>>"
        )

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"User Goal: {requirement}\n\nDEBATE TRANSCRIPT:\n{debate_transcript}")
        ]

        return self.llm.invoke(messages).content
    
    def _generate_final_report(self, state: GlobalAgentState) -> MultiPersonaResponse:
        """Generate the final user-facing answer from the aggregated state."""
        messages = [
            SystemMessage(content= self.instructions_ans),
            AIMessage(content=f'analysis results:{str(state["analysis_result"])}'),
            HumanMessage(content=f'user question: {str(state["requirement"])}'),
        ]
        response_content: str = self.llm.invoke(messages).content

        return MultiPersonaResponse(text=response_content, run_id=state['run_id'], figures=state['analysis_result'].figures)
    
    #! Not in use
    def _summarize_user_request(self, human_req: str) -> str:
        """Summarize the user request with the master prompt."""
        messages = [
            SystemMessage(content= self.instructions_user_req),
            HumanMessage(content=f'user request:{human_req}'),
        ]

        response = self.llm.invoke(messages)
        return response.text
    
    def _log_and_notify(self, message: str, logger: logging.Logger, progress_callback: Union[callable, None] = None, level: str = "info"):
        """
        Logs a message and optionally sends it to the progress callback.
        """
        if level.lower() == "error":
            logger.error(message)
        elif level.lower() == "warning":
            logger.warning(message)
        else:
            logger.info(message)

        if progress_callback:
            progress_callback(message)
    
    def run_request_demo(self,
                    human_input: str, 
                    file_list: List[str], 
                    workspace: SessionWorkspace,
                    progress_callback=None) -> MultiPersonaResponse:
        """Demo pathway: analyze existing figures and produce a final answer."""
        
        final_output = MultiPersonaResponse(
                    run_id="run_12345_abcde",
                    text="## Consolidated Financial Analysis\n\nBased on the multi-perspective review, the company demonstrates strong fundamentals but faces short-term volatility risks.\n\n### Key Findings\n1. **Revenue Growth**: Year-over-year revenue has increased by 15%, driven largely by the new cloud division.\n2. **Risk Factors**: Supply chain disruptions in the semiconductor sector remain a primary concern.\n\n### Conclusion\nThe consensus suggests a **Hold** rating for short-term investors, while long-term value investors may find the current dip an attractive entry point.",
                    perspectives=[
          {
            "role": "The Bull",
            "bias": "Optimistic",
            "icon": "🚀",
            "content": "I see massive upside potential here! The 15% revenue growth is just the beginning. Their new AI product line hasn't even fully hit the market yet, which could double their addressable market size next quarter. The supply chain issues are temporary noise."
          },
          {
            "role": "The Bear",
            "bias": "Pessimistic",
            "icon": "🐻",
            "content": "This growth is unsustainable. That 15% increase is purely inflationary pricing, not volume growth. If you look at the operating margins, they are shrinking. Plus, insider selling has increased by 20% in the last month. This is a trap."
          },
          {
            "role": "The Accountant",
            "bias": "Analytical",
            "icon": "📊",
            "content": "Let's look at the numbers objectively. The current P/E ratio of 25 is slightly above the industry average of 22. Free cash flow is positive but declining quarter-over-quarter. The debt-to-equity ratio is healthy at 0.5, suggesting they are not over-leveraged."
          },
          {
            "role": "The Strategist",
            "bias": "Forward-Looking",
            "icon": "♟️",
            "content": "The crucial factor here is the competitor landscape. While this company is growing, their main rival just secured a major government contract. They need to pivot their R&D spend immediately to stay relevant in the government sector or risk losing market share."
          }
        ],
                    figures=[]
                )
        return final_output
     
        # return FinalAnswer(text="Result.", run_id=workspace.run_id, figures=[PydanticDiagramResult(filename="gold_annual_returns.png", text="Analysis Result")])
        with self._session_logger(workspace) as logger:
            try:
                state = self._initialize_agent_state(workplace=workspace, requirement=human_input, file_list=[])
                diagram_paths = workspace.list_figures()
                state['visualization_paths'] = diagram_paths
                self._log_and_notify('Analysis in Progress', logger=logger, progress_callback=progress_callback)
                final_state = self.analysis_agent.analyze_all_diagrams(state=state, prompt=f'Give insights on these diagrams regarding user request:{state["requirement"]}')
                self._log_and_notify('Fabricating Final Answer', logger=logger, progress_callback=progress_callback)
                final_result = self._synthesis_reports(final_state)
                logger.info(final_result)
                return final_result

            except Exception as e:
                logger.error(f"Run failed: {e}", exc_info=True)
                return MultiPersonaResponse(text=default_config.ERROR_MSG_EXECUTION_FAILED, run_id=workspace.run_id)
    
    def _broadcast_graph_structure(self, send_sse):
        """Call this when the Planner Agent finishes creating the TaskGraph"""
        nodes = []
        edges = []
        
        execution_order: list[str] = self.task_graph.get_execution_order()
        if not execution_order:
            return

        for tid in self.task_graph.nodes:
            node = self.task_graph.nodes[tid]
            # 1. Build Node
            nodes.append({
                "id": node.node_id,
                "label": node.node_name,
                "status": node.status,
                "description": node.instruction,
                # Extract action steps for detailed view
                "sub_steps": [
                    {"id": action.action_id, "desc": action.description}
                    for action in node.action_graph.nodes
                ]
            })
            
            # 2. Build Edges
            for dep in node.dependencies:
                edges.append({"source": dep, "target": node.node_id})

        send_sse({"nodes": nodes, "edges": edges}, "graph_init")

    def run_request(self, 
                    human_input: str, 
                    file_list: List[str],
                    workspace: SessionWorkspace,
                    analyze_only = False,
                    progress_callback=None) -> MultiPersonaResponse:
        """Full workflow: build/refine task graph, execute tasks, analyze diagrams, and craft answer."""
        
        import time
        from copy import deepcopy
        workspace = SessionWorkspace(workspace.sess_id, workspace.run_id)
        
        start_time = time.time()
        log_summary = {
            'sess_id': workspace.sess_id,
            'run_id': workspace.run_id,
            'user_request': human_input,
            'model':{'name':self.llm_config.OPENAI_MODEL,'timeout':self.llm_config.TIMEOUT,'cache':self.llm_config.CACHE,'temperature':self.llm_config.TEMPERATURE,'max_tokens':self.llm_config.MAX_COMPLETION_TOKENS}
        }
        status = default_config.STATUS_FAILED

        initial_state: GlobalAgentState = self._initialize_agent_state(workplace=workspace, requirement=human_input, file_list=file_list)
        workflow_state: GlobalAgentState = deepcopy(initial_state)
        agent_state_version = {'initial':initial_state}

        # Settings from Frontend --
        max_retries_action = self.graph_config.ACTION_GRAPH_MAX_RETRIES
        max_retries_task = self.graph_config.TASK_GRAPH_MAX_RETRIES
        # --

        with self._session_logger(workspace) as logger:
            try:
                
                if not analyze_only:
                    self._load_conversation_history(workspace, logger=logger)
                    self.conversation_history.append(
                        HumanMessage(content=human_input, additional_kwargs={"run_id": workspace.run_id})
                    )
                    file_context: List[Dict[str, Any]] = FileUtils.format_files_for_llm(file_list, workspace.data_dir)
                    
                    previous_state = workspace.load_graph_state()
                    if previous_state:
                        self._log_and_notify("Restoring previous session state...", logger=logger, progress_callback=progress_callback)
                        self.task_graph = TaskGraph.from_dict(previous_state, self.llm_config, self.graph_config)
                        
                        self._log_and_notify('Updating TaskGraph', logger=logger, progress_callback=progress_callback)
                        self.task_graph.refine_plan(file_context=file_context, history=self.conversation_history, refine_instruction=self.task_graph.sys_instructions_refine, max_retries_action=max_retries_action, max_retries_task=max_retries_task)
                    else:
                        self._log_and_notify('Initiating TaskGraph', logger=logger, progress_callback=progress_callback)
                        self.task_graph.generate_plan(human_input=human_input, file_context=file_context, max_retries_task=max_retries_task, max_retries_action=max_retries_action)
                        logger.info(f"TaskGraph created with {len(self.task_graph.nodes)} Nodes")
                    
                    self._broadcast_graph_structure(send_sse=progress_callback)

                    import time
                    time.sleep(0.5)
                    # final_output = MultiPersonaResponse(
                    #     run_id="run_12345_abcde",
                    #     text="## Consolidated Financial Analysis\n\nBased on the multi-perspective review, the company demonstrates strong fundamentals but faces short-term volatility risks.\n\n### Key Findings\n1. **Revenue Growth**: Year-over-year revenue has increased by 15%, driven largely by the new cloud division.\n2. **Risk Factors**: Supply chain disruptions in the semiconductor sector remain a primary concern.\n\n### Conclusion\nThe consensus suggests a **Hold** rating for short-term investors, while long-term value investors may find the current dip an attractive entry point.",
                    #     perspectives=[],
                    #     figures=[])
                    # return final_output
                
                    # Graph Execution
                    workflow_state: GlobalAgentState = self.task_graph.execute_pipeline(
                        initial_state=initial_state,
                        workspace=workspace,
                        progress_callback=progress_callback,
                        stop_on_failure=False
                    )

                # Gathering generated diagrams
                diagram_paths = workspace.list_figures()
                if not diagram_paths:
                    logger.warning("No diagrams found to analyze.")
                    workflow_state['visualization_paths'] = []

                else:
                    workflow_state['visualization_paths'] = diagram_paths
                    logger.info(f"{self.__class__.__name__} - Found {len(diagram_paths)} diagrams to analyze.")
                    
                agent_state_version['after workflow'] = workflow_state
                
                # Diagram Selection for Analysis
                selected_diagrams: List[str] = self.analysis_agent.select_key_diagrams(state=workflow_state, max_items=5)
                log_summary['selected_diagrams'] = selected_diagrams
                logger.info(f'Selcted and analyzing {len(selected_diagrams)} diagrams...')

                # Analysis: Multi-Personal Analysis Debate
                self._log_and_notify(f"Consulting {len(self.personas)} Personas...", logger=logger, progress_callback=progress_callback)
                collected_insights: List[PersonaResponse] = self.analysis_agent.analyze_diagrams_with_all_personas(workflow_state=workflow_state, diagram_paths=selected_diagrams, personas=self.personas, logger=logger)
                evaluated_insights: List[AgentEvaluationResult] = self.bias_evaluator.evaluate_batch(agent_responses=collected_insights)

                # Store Agent State
                final_state: GlobalAgentState = deepcopy(workflow_state)
                final_state['analysis_result'] = evaluated_insights

                # Neutral Report Synthesis
                self._log_and_notify('Synthesizing Neutral Report...', logger=logger, progress_callback=progress_callback)
                neutral_report: str = self._synthesize_reports(human_input, collected_insights)
                final_state['neutral_report'] = neutral_report
                agent_state_version['final'] = final_state

                self.conversation_history.append(
                    AIMessage(content=neutral_report, additional_kwargs={"run_id": workspace.run_id})
                )

                final_output = MultiPersonaResponse(
                    run_id=workspace.run_id,
                    text=neutral_report,
                    perspectives=evaluated_insights,
                    figures=final_state.get('visualization_paths', [])
                )

                status = default_config.STATUS_SUCCESS
                return final_output
            
            except Exception as e:
                c_logger.error(f"Run failed: {e}")
                logger.error(f"Run failed: {e}", exc_info=True)
                raise RuntimeError(default_config.ERROR_MSG_EXECUTION_FAILED)
            
            finally:
                self._save_conversation_history(workspace, logger=logger)
                workspace.save_graph_state(self.task_graph.to_dict())
                self.task_graph.save_code(sess_id=initial_state['sess_id'], run_id=initial_state['run_id'], verbose=True)

                logger.info('Saved TaskGraph and Conversation History')

                # Generate Request Summary
                duration_sec = time.time() - start_time
                log_summary['status'] = status
                log_summary['duration_sec'] = round(duration_sec, 2)
                c_logger.info(
                    f"Finished request",
                    extra=log_summary
                )
                workspace.save_json(data=agent_state_version)
                logger.info(f"Closing log handler for Run ID {workspace.run_id}")


def get_master_agent(config: AppSettings):
    """Factory helper to construct a MasterAgent with the default model."""
    return MasterAgent(config=config)