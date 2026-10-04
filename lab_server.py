"""Exposes the agent tools as an MCP server so agents can call them from any Python env.

    python lab_server.py                              one stdio server per agent (default)
    python lab_server.py --http --max-passes 5000     one shared server over HTTP at http://127.0.0.1:8765/mcp
"""
import argparse

from mcp.server.fastmcp import FastMCP

import agent_tools
import lab_tools
import research_record

mcp = FastMCP("lab")

for fn in (
    agent_tools.make_dataset,
    agent_tools.run_baseline,
    agent_tools.patch,
    agent_tools.ablate,
    agent_tools.path_patch,
    agent_tools.path_patch_sweep,
    agent_tools.faithfulness,
    agent_tools.score_vs_truth,
    agent_tools.budget,
    agent_tools.search_literature,
    agent_tools.present_claim,
    research_record.log_event,
    research_record.read_record,
):
    mcp.tool()(fn)  # name, inputs and description come from each function

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--http", action="store_true", help="serve over streamable HTTP so every agent shares one process")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--max-passes", type=int, default=None, help="forward-pass budget for this run")
    args = parser.parse_args()
    if args.max_passes is not None:
        lab_tools.budget(reset=True, max_passes=args.max_passes)
    if args.http:
        mcp.settings.host = "127.0.0.1"
        mcp.settings.port = args.port
        mcp.run(transport="streamable-http")
    else:
        mcp.run()  # talks over stdin/stdout
