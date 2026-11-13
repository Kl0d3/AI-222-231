
import asyncio
import threading

from src.mcp.client_manager import MCPClientManager


class MCPBackgroundManager:

    def __init__(self):

        self.manager = None

        self.loop = None
        self.thread = None

        self.loop_ready = threading.Event()
        self.initialization_complete = threading.Event()

        self.shutdown_event = None
        self.lifecycle_future = None

        self.initialization_error = None

        self.started = False

    # ====================================================
    # Background event loop
    # ====================================================

    def _run_event_loop(self):

        self.loop = asyncio.new_event_loop()

        asyncio.set_event_loop(self.loop)

        self.loop_ready.set()

        try:

            self.loop.run_forever()

        finally:

            pending_tasks = asyncio.all_tasks(self.loop)

            for task in pending_tasks:
                task.cancel()

            if pending_tasks:

                self.loop.run_until_complete(
                    asyncio.gather(
                        *pending_tasks,
                        return_exceptions=True
                    )
                )

            self.loop.close()

            print("\n[MCP BACKGROUND] Event loop stopped.")

    # ====================================================
    # MCP lifecycle
    # ====================================================

    async def _manager_lifecycle(self):

        # This task owns the complete MCP connection lifecycle.
        self.shutdown_event = asyncio.Event()

        self.manager = MCPClientManager()

        try:

            print(
                "\n[MCP BACKGROUND] Connecting to MCP servers..."
            )

            await self.manager.connect_all()

            print(
                "\n[MCP BACKGROUND] Discovering MCP tools..."
            )

            await self.manager.discover_tools()

            print(
                "\n[MCP BACKGROUND] Initialization complete."
            )

            self.started = True

            self.initialization_complete.set()

            # Keep the lifecycle task alive until shutdown.
            await self.shutdown_event.wait()

        except Exception as error:

            self.initialization_error = error

            self.initialization_complete.set()

            raise

        finally:

            if self.manager is not None:

                print(
                    "\n[MCP BACKGROUND] Closing MCP connections..."
                )

                try:

                    # Important:
                    # Connections are closed in the same task
                    # that originally entered their contexts.
                    await self.manager.close()

                finally:

                    self.manager = None

            self.started = False

    # ====================================================
    # Start background manager
    # ====================================================

    def start(self, timeout=30):

        if self.started:

            print(
                "\n[MCP BACKGROUND] Already running."
            )

            return

        if self.thread is not None and self.thread.is_alive():

            raise RuntimeError(
                "MCP background thread is already running."
            )

        self.loop_ready.clear()
        self.initialization_complete.clear()

        self.initialization_error = None

        self.thread = threading.Thread(
            target=self._run_event_loop,
            name="MCP-Event-Loop",
            daemon=True
        )

        self.thread.start()

        if not self.loop_ready.wait(timeout=10):

            raise TimeoutError(
                "MCP background event loop failed to start."
            )

        self.lifecycle_future = (
            asyncio.run_coroutine_threadsafe(
                self._manager_lifecycle(),
                self.loop
            )
        )

        if not self.initialization_complete.wait(timeout=timeout):

            self.close()

            raise TimeoutError(
                "MCP server initialization timed out."
            )

        if self.initialization_error is not None:

            error = self.initialization_error

            self.close()

            raise RuntimeError(
                "MCP background manager initialization failed."
            ) from error

        print(
            "\n[MCP BACKGROUND] Manager started successfully."
        )

    # ====================================================
    # Retrieve Gemini tool declarations
    # ====================================================

    async def _get_gemini_tools(self):

        if self.manager is None:

            raise RuntimeError(
                "MCP manager is not initialized."
            )

        return self.manager.get_gemini_tools()

    def get_gemini_tools(self):

        self._ensure_running()

        future = asyncio.run_coroutine_threadsafe(
            self._get_gemini_tools(),
            self.loop
        )

        return future.result()

    # ====================================================
    # Execute MCP tool
    # ====================================================

    async def _call_tool(self, tool_name, arguments=None):

        if self.manager is None:

            raise RuntimeError(
                "MCP manager is not initialized."
            )

        return await self.manager.call_tool(
            tool_name=tool_name,
            arguments=arguments
        )

    def call_tool(
        self,
        tool_name,
        arguments=None,
        timeout=None
    ):

        self._ensure_running()

        future = asyncio.run_coroutine_threadsafe(
            self._call_tool(
                tool_name=tool_name,
                arguments=arguments
            ),
            self.loop
        )

        return future.result(timeout=timeout)

    # ====================================================
    # Validate manager state
    # ====================================================

    def _ensure_running(self):

        if not self.started:

            raise RuntimeError(
                "MCP background manager is not running. "
                "Call start() first."
            )

        if self.loop is None or not self.loop.is_running():

            raise RuntimeError(
                "MCP background event loop is not running."
            )

    # ====================================================
    # Shutdown
    # ====================================================

    def close(self):

        if self.loop is None:

            return

        try:

            if (
                self.lifecycle_future is not None
                and not self.lifecycle_future.done()
            ):

                print(
                    "\n[MCP BACKGROUND] Requesting shutdown..."
                )

                # Signal the lifecycle task on its own event loop.
                self.loop.call_soon_threadsafe(
                    self.shutdown_event.set
                )

                # Wait for the lifecycle task to close all
                # MCP connections before stopping the loop.
                self.lifecycle_future.result(
                    timeout=30
                )

        finally:

            if self.loop is not None and self.loop.is_running():

                self.loop.call_soon_threadsafe(
                    self.loop.stop
                )

            if self.thread is not None and self.thread.is_alive():

                self.thread.join(timeout=10)

                if self.thread.is_alive():

                    raise TimeoutError(
                        "MCP background thread did not stop."
                    )

            self.started = False

            self.manager = None
            self.loop = None
            self.thread = None

            self.lifecycle_future = None
            self.shutdown_event = None

            self.loop_ready.clear()
            self.initialization_complete.clear()

            print(
                "\n[MCP BACKGROUND] Manager shutdown complete."
            )
