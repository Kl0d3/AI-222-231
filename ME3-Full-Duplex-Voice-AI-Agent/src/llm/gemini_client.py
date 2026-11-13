
import os
import json
import copy
import threading
from pathlib import Path

from dotenv import load_dotenv
from google import genai


PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")


class GeminiClient:

    SYSTEM_INSTRUCTION = """
You are a natural, conversational voice assistant.

Follow these communication guidelines:

1. Speak naturally, as if having a real conversation.
2. Keep responses concise and easy to understand.
3. Prefer short paragraphs over long explanations.
4. Avoid unnecessary introductions and repetitive conclusions.
5. Understand follow-up questions using the conversation history.
6. Do not ask for clarification when the conversation context
   already makes the user's meaning reasonably clear.
7. If the user asks a simple question, answer it directly.
8. Avoid Markdown formatting, tables, and bullet lists unless
   the user explicitly requests them.
9. Explain technical concepts in clear, beginner-friendly language.
10. Do not repeat information the user already understands
    unless it is necessary.
11. Use available tools when they are needed to retrieve current
    information or perform an action.
12. Never invent tool results or claim an action succeeded
    unless the tool reports success.
13. If a tool fails, explain the failure honestly and naturally.

WEBSITE RESPONSE GUIDELINES:

14. When a website has been successfully created and published,
    respond with only:
    "Your website has been created and published."

15. When a website has been successfully created locally but
    not published, respond with only:
    "Your website has been created."

16. Do not provide feature summaries, implementation details,
    lists of included content, tool descriptions, or website URLs
    unless the user explicitly asks for them.

17. If website creation or publishing fails, briefly explain
    what failed. Never claim that an unsuccessful operation
    was completed.

IMPORTANT ACTION SAFETY:

18. Creating, deleting, or completing Google Calendar events
    and Google Tasks requires explicit user confirmation.
19. Deleting a local website project also requires explicit
    user confirmation.
20. When a protected operation is requested, wait for the
    application to obtain confirmation before executing it.
21. If the application reports that an operation is awaiting
    confirmation, do not claim that it has been executed.
22. Never attempt to bypass the application's confirmation
    mechanism by calling the same operation again.
23. Clearly summarize the proposed operation when asking
    the user for confirmation.
24. Publishing a new website or updating an existing
    agent-managed website does not require an additional
    confirmation after the user has requested that action.
25. Never delete a GitHub repository or change repository
    visibility without explicit user confirmation.

Your responses will be converted into speech, so prioritize
clarity, naturalness, and conversational flow.
"""

    MAX_TOOL_ROUNDS = 10

    PROTECTED_TOOLS = {
        "create_calendar_event",
        "delete_calendar_event",
        "create_task",
        "complete_task",
        "delete_task",
        "delete_website_project",
    }

    def __init__(
        self,
        model=None,
        mcp_manager=None,
        feedback_callback=None
    ):

        api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY was not found. "
                "Check your project's .env file."
            )

        self.model = model or os.getenv(
            "GEMINI_MODEL",
            "gemini-3.5-flash-lite"
        )

        self.client = genai.Client(api_key=api_key)

        self.mcp_manager = mcp_manager

        # Optional callback for think-aloud and progress feedback.
        self.feedback_callback = feedback_callback

        self.previous_interaction_id = None
        self.pending_interaction_id = None

        self.pending_action_lock = threading.Lock()
        self.pending_action = None

        print(f"Gemini client initialized using {self.model}")

        if self.mcp_manager is not None:
            print("[GEMINI] MCP manager connected.")

        if self.feedback_callback is not None:
            print("[GEMINI] Speech feedback callback connected.")

    # ====================================================
    # Feedback notifications
    # ====================================================

    def notify_feedback(self, event, tool_name=None):

        if self.feedback_callback is None:
            return

        try:

            self.feedback_callback(
                event,
                tool_name
            )

        except Exception as error:

            # Feedback must never break Gemini processing.
            print(
                "\n[GEMINI FEEDBACK WARNING]",
                error
            )

    # ====================================================
    # Retrieve MCP tool declarations
    # ====================================================

    def get_available_tools(self):

        if self.mcp_manager is None:
            return []

        tools = self.mcp_manager.get_gemini_tools()

        print(
            f"\n[GEMINI] Available MCP tools: {len(tools)}"
        )

        return tools

    # ====================================================
    # Stage protected operations
    # ====================================================

    def stage_protected_action(self, function_call):

        tool_name = function_call.name
        arguments = function_call.arguments

        if arguments is None:
            arguments = {}

        elif isinstance(arguments, str):

            try:
                arguments = json.loads(arguments)

            except json.JSONDecodeError:

                return {
                    "success": False,
                    "tool": tool_name,
                    "error": "Invalid JSON function arguments."
                }

        if not isinstance(arguments, dict):

            return {
                "success": False,
                "tool": tool_name,
                "error": "Function arguments must be an object."
            }

        staged_arguments = copy.deepcopy(arguments)

        with self.pending_action_lock:

            if self.pending_action is not None:

                return {
                    "success": False,
                    "tool": tool_name,
                    "requires_confirmation": True,
                    "error": (
                        "Another protected operation is already "
                        "awaiting user confirmation. Do not execute "
                        "this additional operation."
                    )
                }

            self.pending_action = {
                "tool": tool_name,
                "arguments": staged_arguments,
            }

        print(
            f"\n[GEMINI] Protected operation staged: {tool_name}"
        )

        print(
            "[GEMINI] Waiting for explicit user confirmation."
        )

        return {
            "success": False,
            "tool": tool_name,
            "requires_confirmation": True,
            "message": (
                "The requested operation has been prepared "
                "but NOT executed. Explicit user confirmation "
                "is required before execution."
            ),
            "proposed_operation": {
                "tool": tool_name,
                "arguments": copy.deepcopy(staged_arguments),
            }
        }

    # ====================================================
    # Execute a confirmed protected operation
    # ====================================================

    def execute_pending_action(self):

        with self.pending_action_lock:

            if self.pending_action is None:

                raise RuntimeError(
                    "There is no protected operation awaiting confirmation."
                )

            action = self.pending_action
            self.pending_action = None

        tool_name = action["tool"]
        arguments = action["arguments"]

        print(
            f"\n[GEMINI] Confirmed operation claimed: {tool_name}"
        )

        if tool_name not in self.PROTECTED_TOOLS:

            raise RuntimeError(
                "The pending operation is not an authorized "
                "protected tool."
            )

        if self.mcp_manager is None:

            raise RuntimeError(
                "MCP manager is not available."
            )

        print(
            f"\n[GEMINI] Executing confirmed operation: {tool_name}"
        )

        self.notify_feedback(
            "tool_started",
            tool_name
        )

        result = self.mcp_manager.call_tool(
            tool_name=tool_name,
            arguments=arguments
        )

        return result

    # ====================================================
    # Cancel a pending protected operation
    # ====================================================

    def cancel_pending_action(self):

        with self.pending_action_lock:

            if self.pending_action is None:
                return

            tool_name = self.pending_action["tool"]

            self.pending_action = None

        print(
            f"\n[GEMINI] Pending operation cancelled: {tool_name}"
        )

    # ====================================================
    # Execute ordinary MCP function calls
    # ====================================================

    def execute_function_call(self, function_call):

        tool_name = function_call.name
        arguments = function_call.arguments

        print(
            f"\n[GEMINI TOOL CALL] {tool_name}"
        )

        # Protected operations are only staged here.
        # They are not executed until confirmation.
        if tool_name in self.PROTECTED_TOOLS:

            return self.stage_protected_action(
                function_call
            )

        if self.mcp_manager is None:

            return {
                "success": False,
                "tool": tool_name,
                "error": "MCP manager is not available."
            }

        if arguments is None:
            arguments = {}

        elif isinstance(arguments, str):

            try:
                arguments = json.loads(arguments)

            except json.JSONDecodeError:

                return {
                    "success": False,
                    "tool": tool_name,
                    "error": "Invalid JSON function arguments."
                }

        if not isinstance(arguments, dict):

            return {
                "success": False,
                "tool": tool_name,
                "error": "Function arguments must be an object."
            }

        try:

            # Notify immediately before actual MCP execution.
            self.notify_feedback(
                "tool_started",
                tool_name
            )

            result = self.mcp_manager.call_tool(
                tool_name=tool_name,
                arguments=arguments
            )

            return result

        except Exception as error:

            print(
                f"\n[GEMINI TOOL ERROR] {tool_name}: {error}"
            )

            return {
                "success": False,
                "tool": tool_name,
                "error": str(error)
            }

    # ====================================================
    # Process Gemini function calls
    # ====================================================

    def process_function_calls(self, interaction, tools):

        current_interaction = interaction

        for round_number in range(1, self.MAX_TOOL_ROUNDS + 1):

            function_calls = [
                step
                for step in (current_interaction.steps or [])
                if step.type == "function_call"
            ]

            if not function_calls:
                break

            print(
                f"\n[GEMINI] Function-calling round "
                f"{round_number}"
            )

            function_results = []

            for function_call in function_calls:

                if not function_call.id or not function_call.name:

                    raise RuntimeError(
                        "Gemini returned a function call "
                        "without a valid ID or name."
                    )

                result = self.execute_function_call(
                    function_call
                )

                result_json = json.dumps(
                    result,
                    ensure_ascii=False,
                    default=str
                )

                function_results.append({
                    "type": "function_result",
                    "name": function_call.name,
                    "call_id": function_call.id,
                    "result": [
                        {
                            "type": "text",
                            "text": result_json
                        }
                    ]
                })

            print(
                f"\n[GEMINI] Returning "
                f"{len(function_results)} tool result(s)."
            )

            current_interaction = (
                self.client.interactions.create(
                    model=self.model,
                    previous_interaction_id=current_interaction.id,
                    input=function_results,
                    tools=tools
                )
            )

        else:

            raise RuntimeError(
                "Gemini exceeded the maximum number of "
                "function-calling rounds."
            )

        return current_interaction

    # ====================================================
    # Generate response
    # ====================================================

    def generate_response(self, prompt, allow_tools=True):

        if not prompt or not prompt.strip():

            raise ValueError(
                "Prompt cannot be empty."
            )

        if self.pending_interaction_id is not None:

            raise RuntimeError(
                "A previous Gemini response is still pending "
                "commit or discard."
            )

        request_parameters = {
            "model": self.model,
            "input": prompt,
            "system_instruction": self.SYSTEM_INSTRUCTION
        }

        if allow_tools:

            tools = self.get_available_tools()

            if tools:
                request_parameters["tools"] = tools

        else:

            print(
                "\n[GEMINI] Tool calling disabled for this response."
            )

        if self.previous_interaction_id is not None:

            request_parameters["previous_interaction_id"] = (
                self.previous_interaction_id
            )

        print("\n[GEMINI] Generating response...")

        # Notify the feedback manager that Gemini processing
        # is beginning. No additional Gemini request is made.
        self.notify_feedback(
            "processing_started"
        )

        response = self.client.interactions.create(
            **request_parameters
        )

        if allow_tools and request_parameters.get("tools"):

            response = self.process_function_calls(
                interaction=response,
                tools=request_parameters["tools"]
            )

        output_text = response.output_text

        if not output_text or not output_text.strip():

            raise RuntimeError(
                "Gemini returned an empty response."
            )

        self.pending_interaction_id = response.id

        return output_text

    # ====================================================
    # Commit generated response
    # ====================================================

    def commit_pending_interaction(self):

        if self.pending_interaction_id is None:
            return

        self.previous_interaction_id = (
            self.pending_interaction_id
        )

        self.pending_interaction_id = None

        print(
            "\n[GEMINI] Conversation interaction committed."
        )

    # ====================================================
    # Discard generated response
    # ====================================================

    def discard_pending_interaction(self):

        if self.pending_interaction_id is None:
            return

        self.pending_interaction_id = None

        print(
            "\n[GEMINI] Pending interaction discarded."
        )

    # ====================================================
    # Shutdown
    # ====================================================

    def close(self):

        self.cancel_pending_action()

        self.client.close()
