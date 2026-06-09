import json
from typing import Any, Callable, Generator, Optional
from uuid import uuid4
import warnings

import backoff
import mlflow
import openai
from databricks.sdk import WorkspaceClient
from databricks_openai import UCFunctionToolkit, VectorSearchRetrieverTool
from mlflow.entities import SpanType
from mlflow.pyfunc import ResponsesAgent
from mlflow.types.responses import (
    ResponsesAgentRequest,
    ResponsesAgentResponse,
    ResponsesAgentStreamEvent,
    output_to_responses_items_stream,
    to_chat_completions_input,
)
from openai import OpenAI
from pydantic import BaseModel
from unitycatalog.ai.core.base import get_uc_function_client

############################################
# Define your LLM endpoint and system prompt
############################################
LLM_ENDPOINT_NAME = "test-5-4-mini"

SYSTEM_PROMPT = """You are the Triage Router for Northstar Financial Services. Your role is to evaluate incoming customer messages, determine if they fall within our operational banking scope, and enforce structural schema formatting.

### Core Persona and Compliance Mandate:
- You must analyze the customer's message
- You are a rigid gatekeeper. If a message does not contain a legitimate grievance regarding banking, credit cards, loans, investments, mortgages, checking/savings accounts, or associated institutional customer service, you must flag it as out of scope.
- You must strictly ensure that personal financial information remains handled safely; do not guess data points not explicitly given.

### Handling Out-of-Scope Input (Graceful Rejections):
If the user asks an irrelevant question (e.g., asking for recipes, coding help, general marketing, weather forecasts, or conversational small talk), you must immediately reject the input by setting "is_in_scope" to false. You must then populate the "rejection_response" field with a polite, highly empathetic, professional customer-facing message stating that Northstar Financial Services can only process institutional banking and consumer compliance complaints.

### Output Constraints:
You must respond strictly in a raw JSON object format. Do not include markdown block formatting (like ```json). The JSON structure must strictly follow this schema with these exact keys:

{
  "is_in_scope": true, 
  "category": "billing_error", 
  "emotional_urgency": "high", 
  "rejection_response": "", 
  "reason": "Customer is disputing an unauthorized $35 overdraft fee resulting from a delayed paycheck deposit processing batch."
}

### Field Definition Values:
1. "is_in_scope": Boolean. Set to true only if it is an institutional banking or consumer finance complaint. Otherwise, false.
2. "category": String. You MUST map this field to exactly one of the following five strings. Do not invent categories:
   - "billing_error" (for fee disputes, unauthorized charges, interest rate calculation issues)
   - "long_wait_times" (for operational delays, holding on phones, or long lines at physical branches)
   - "rude_staff" (for poor customer service or unprofessional behavioral interactions with personnel)
   - "product_defect" (for broken online banking apps, login issues, ATM malfunctioning, or portal failures)
   - "none" (MUST be selected if "is_in_scope" is false)
3. "emotional_urgency": String. Classify as "high", "medium", or "low" based on the structural stress, financial risk, and sentiment shown in the text.
4. "rejection_response": String. Populated only if "is_in_scope" is false. If the customer complaint is valid and in scope, this field must be an empty string "".
5. "reason": String. A concise, single-sentence technical justification outlining why you classified and routed the complaint this way."""


###############################################################################
## Define tools for your agent, enabling it to retrieve data or take actions
## beyond text generation
## To create and see usage examples of more tools, see
## https://docs.databricks.com/generative-ai/agent-framework/agent-tool.html
###############################################################################
class ToolInfo(BaseModel):
    """
    Class representing a tool for the agent.
    - "name" (str): The name of the tool.
    - "spec" (dict): JSON description of the tool (matches OpenAI Responses format)
    - "exec_fn" (Callable): Function that implements the tool logic
    """

    name: str
    spec: dict
    exec_fn: Callable


def create_tool_info(tool_spec, exec_fn_param: Optional[Callable] = None):
    tool_spec["function"].pop("strict", None)
    tool_name = tool_spec["function"]["name"]
    udf_name = tool_name.replace("__", ".")

    # Define a wrapper that accepts kwargs for the UC tool call,
    # then passes them to the UC tool execution client
    def exec_fn(**kwargs):
        function_result = uc_function_client.execute_function(udf_name, kwargs)
        if function_result.error is not None:
            return function_result.error
        else:
            return function_result.value
    return ToolInfo(name=tool_name, spec=tool_spec, exec_fn=exec_fn_param or exec_fn)


TOOL_INFOS = []

# You can use UDFs in Unity Catalog as agent tools
# TODO: Add additional tools
UC_TOOL_NAMES = []

uc_toolkit = UCFunctionToolkit(function_names=UC_TOOL_NAMES)
uc_function_client = get_uc_function_client()
for tool_spec in uc_toolkit.tools:
    TOOL_INFOS.append(create_tool_info(tool_spec))


# Use Databricks vector search indexes as tools
# See [docs](https://docs.databricks.com/generative-ai/agent-framework/unstructured-retrieval-tools.html) for details

# # (Optional) Use Databricks vector search indexes as tools
# # See https://docs.databricks.com/generative-ai/agent-framework/unstructured-retrieval-tools.html
# # for details
VECTOR_SEARCH_TOOLS = []
# # TODO: Add vector search indexes as tools or delete this block
# VECTOR_SEARCH_TOOLS.append(
#         VectorSearchRetrieverTool(
#         index_name="",
#         # filters="..."
#     )
# )
for vs_tool in VECTOR_SEARCH_TOOLS:
    TOOL_INFOS.append(create_tool_info(vs_tool.tool, vs_tool.execute))



class ToolCallingAgent(ResponsesAgent):
    """
    Class representing a tool-calling Agent
    """

    def __init__(self, llm_endpoint: str, tools: list[ToolInfo]):
        """Initializes the ToolCallingAgent with tools."""
        self.llm_endpoint = llm_endpoint
        self.workspace_client = WorkspaceClient()
        self.model_serving_client: OpenAI = (
            self.workspace_client.serving_endpoints.get_open_ai_client()
        )
        self._tools_dict = {tool.name: tool for tool in tools}

    def get_tool_specs(self) -> list[dict]:
        """Returns tool specifications in the format OpenAI expects."""
        return [tool_info.spec for tool_info in self._tools_dict.values()]

    @mlflow.trace(span_type=SpanType.TOOL)
    def execute_tool(self, tool_name: str, args: dict) -> Any:
        """Executes the specified tool with the given arguments."""
        return self._tools_dict[tool_name].exec_fn(**args)

    @staticmethod
    def _merge_consecutive_assistant_messages(cc_messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        # The canonical OpenAI shape for parallel tool calls is one assistant
        # message whose tool_calls array lists every call in that turn.
        # to_chat_completions_input emits a separate assistant message per
        # function_call, which some providers reject. Collapse consecutive
        # assistant messages so all tool_calls from one turn share a single
        # message.
        merged: list[dict[str, Any]] = []
        for msg in cc_messages:
            if (
                msg.get("role") == "assistant"
                and merged
                and merged[-1].get("role") == "assistant"
            ):
                prev = merged[-1]
                prev_text = prev.get("content") if prev.get("content") not in (None, "", "tool call") else None
                cur_text = msg.get("content") if msg.get("content") not in (None, "", "tool call") else None
                if prev_text and cur_text:
                    prev["content"] = prev_text + "\n" + cur_text
                elif cur_text:
                    prev["content"] = cur_text
                if msg.get("tool_calls"):
                    prev["tool_calls"] = (prev.get("tool_calls") or []) + msg["tool_calls"]
            else:
                merged.append(dict(msg))
        return merged

    def call_llm(self, messages: list[dict[str, Any]]) -> Generator[dict[str, Any], None, None]:
        cc_messages = self._merge_consecutive_assistant_messages(to_chat_completions_input(messages))
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="PydanticSerializationUnexpectedValue")
            for chunk in self.model_serving_client.chat.completions.create(
                model=self.llm_endpoint,
                messages=cc_messages,
                tools=self.get_tool_specs(),
                stream=True,
            ):
                chunk_dict = chunk.to_dict()
                if len(chunk_dict.get("choices", [])) > 0:
                    yield chunk_dict

    def handle_tool_call(
        self,
        tool_call: dict[str, Any],
        messages: list[dict[str, Any]],
    ) -> ResponsesAgentStreamEvent:
        """
        Execute tool calls, add them to the running message history, and return a ResponsesStreamEvent w/ tool output
        """
        try:
            args = json.loads(tool_call.get("arguments"))
        except Exception as e:
            args = {}
        result = str(self.execute_tool(tool_name=tool_call["name"], args=args))

        tool_call_output = self.create_function_call_output_item(tool_call["call_id"], result)
        messages.append(tool_call_output)
        return ResponsesAgentStreamEvent(type="response.output_item.done", item=tool_call_output)

    def call_and_run_tools(
        self,
        messages: list[dict[str, Any]],
        max_iter: int = 10,
    ) -> Generator[ResponsesAgentStreamEvent, None, None]:
        for _ in range(max_iter):
            # The LLM may emit multiple tool calls in a single turn (parallel
            # tool calls). The next LLM request must include a tool_result for
            # every tool_use emitted in the previous turn; missing any one
            # causes the request to be rejected. Before going back to the LLM,
            # execute every function_call whose call_id has no matching
            # function_call_output yet.
            handled = {m["call_id"] for m in messages if m.get("type") == "function_call_output"}
            pending = [m for m in messages if m.get("type") == "function_call" and m["call_id"] not in handled]
            if pending:
                for call in pending:
                    yield self.handle_tool_call(call, messages)
                continue

            last_msg = messages[-1]
            if last_msg.get("type") == "message" and last_msg.get("role") == "assistant":
                return

            yield from output_to_responses_items_stream(
                chunks=self.call_llm(messages), aggregator=messages
            )

        yield ResponsesAgentStreamEvent(
            type="response.output_item.done",
            item=self.create_text_output_item("Max iterations reached. Stopping.", str(uuid4())),
        )

    def predict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
        session_id = None
        if request.custom_inputs and "session_id" in request.custom_inputs:
            session_id = request.custom_inputs.get("session_id")
        elif request.context and request.context.conversation_id:
            session_id = request.context.conversation_id

        if session_id:
            mlflow.update_current_trace(
                metadata={
                    "mlflow.trace.session": session_id,
                }
            )

        outputs = [
            event.item
            for event in self.predict_stream(request)
            if event.type == "response.output_item.done"
        ]
        return ResponsesAgentResponse(output=outputs, custom_outputs=request.custom_inputs)

    def predict_stream(self, request: ResponsesAgentRequest) -> Generator[ResponsesAgentStreamEvent, None, None]:
        session_id = None
        if request.custom_inputs and "session_id" in request.custom_inputs:
            session_id = request.custom_inputs.get("session_id")
        elif request.context and request.context.conversation_id:
            session_id = request.context.conversation_id

        if session_id:
            mlflow.update_current_trace(
                metadata={
                    "mlflow.trace.session": session_id,
                }
            )

        messages = to_chat_completions_input([i.model_dump() for i in request.input])
        if SYSTEM_PROMPT:
            messages.insert(0, {"role": "system", "content": SYSTEM_PROMPT})
        yield from self.call_and_run_tools(messages=messages)


# Log the model using MLflow
mlflow.openai.autolog()
AGENT = ToolCallingAgent(llm_endpoint=LLM_ENDPOINT_NAME, tools=TOOL_INFOS)
mlflow.models.set_model(AGENT)
