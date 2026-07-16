from langgraph.graph import StateGraph, END
from app.agent.state import AgentState
from app.agent.nodes import (
    safety_check_node, exif_extract_node, vision_analyze_node,
    clue_extract_node, react_loop_node, result_synthesize_node,
)


def should_continue_react(state: AgentState) -> str:
    """Determine whether to continue the ReAct loop or finish."""
    if state.get("error"):
        return "result_synthesize"

    result = state.get("result", {})
    if result and ("address" in result or result.get("_action") == "final_answer"):
        return "result_synthesize"

    loop_count = state.get("loop_count", 0)
    try:
        from app.config import get_settings
        settings_loops = get_settings().max_react_loops
    except Exception:
        settings_loops = 10

    if loop_count >= settings_loops:
        return "result_synthesize"

    return "react_loop"


def should_continue_after_safety(state: AgentState) -> str:
    if state.get("safety_passed"):
        return "exif_extract"
    return END


def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("safety_check", safety_check_node)
    graph.add_node("exif_extract", exif_extract_node)
    graph.add_node("vision_analyze", vision_analyze_node)
    graph.add_node("clue_extract", clue_extract_node)
    graph.add_node("react_loop", react_loop_node)
    graph.add_node("result_synthesize", result_synthesize_node)

    graph.set_entry_point("safety_check")

    graph.add_conditional_edges("safety_check", should_continue_after_safety, {
        "exif_extract": "exif_extract",
        END: END,
    })

    graph.add_edge("exif_extract", "vision_analyze")
    graph.add_edge("vision_analyze", "clue_extract")
    graph.add_edge("clue_extract", "react_loop")

    graph.add_conditional_edges("react_loop", should_continue_react, {
        "react_loop": "react_loop",
        "result_synthesize": "result_synthesize",
    })

    graph.add_edge("result_synthesize", END)

    return graph.compile()
