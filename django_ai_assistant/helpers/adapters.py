import asyncio
import inspect
from typing import Any, Dict, List

from langchain_core.messages import ToolMessage
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool


class ToolNode(Runnable):
    def __init__(self, tools: List[BaseTool]):
        self.tools = {t.name: t for t in tools}

    def invoke(self, state: Dict[str, Any], config=None) -> Dict[str, Any]:
        tool_call = self._extract_tool_call(state)
        if tool_call is None:
            return state  # silent no-op

        name = tool_call["name"]
        args = tool_call.get("args", {})
        if name not in self.tools:
            raise ValueError(f"Unknown tool '{name}' requested by model.")

        tool = self.tools[name]

        # Sync run (force sync even if tool is async)
        if inspect.iscoroutinefunction(tool.invoke):
            output = asyncio.run(tool.invoke(args))
        else:
            output = tool.invoke(args)

        tool_msg = ToolMessage(
            content=str(output),
            name=name,
            tool_call_id=tool_call.get("id"),
        )

        messages = [*list(state.get("messages", [])), tool_msg]
        return {**state, "messages": messages}

    async def ainvoke(self, state: Dict[str, Any], config=None) -> Dict[str, Any]:
        tool_call = self._extract_tool_call(state)
        if tool_call is None:
            return state

        name = tool_call["name"]
        args = tool_call.get("args", {})
        if name not in self.tools:
            raise ValueError(f"Unknown tool '{name}' requested by model.")

        tool = self.tools[name]

        # Async run (wrap sync tools)
        if inspect.iscoroutinefunction(tool.invoke):
            output = await tool.invoke(args)
        else:
            output = await asyncio.to_thread(tool.invoke, args)

        tool_msg = ToolMessage(
            content=str(output),
            name=name,
            tool_call_id=tool_call.get("id"),
        )

        messages = [*list(state.get("messages", [])), tool_msg]
        return {**state, "messages": messages}

    def batch(self, states: List[Dict[str, Any]], config=None):
        return [self.invoke(s, config) for s in states]

    async def abatch(self, states: List[Dict[str, Any]], config=None):
        return [await self.ainvoke(s, config) for s in states]

    def _extract_tool_call(self, state: Dict[str, Any]):
        # agent_output style
        agent_output = state.get("agent_output")
        if agent_output and getattr(agent_output, "tool_calls", None):
            calls = agent_output.tool_calls
            if calls:
                return calls[0]

        # message-based style
        msgs = state.get("messages", [])
        if msgs:
            last = msgs[-1]
            # LCEL style
            if hasattr(last, "tool_calls") and last.tool_calls:
                return last.tool_calls[0]
            # raw dict fallback
            if isinstance(last, dict) and "tool_calls" in last:
                tcs = last["tool_calls"]
                if isinstance(tcs, list) and tcs:
                    return tcs[0]

        return None
