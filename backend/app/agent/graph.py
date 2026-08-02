from langgraph.graph import StateGraph, END
from app.agent.state import AgentState
from app.agent.nodes import (
    safety_check_node, independent_signals_node,
    vision_detail_node, clue_extract_node, ocr_fusion_node, anchor_search_node,
    search_strategy_node, react_loop_node,
    result_synthesize_node, adversarial_verify_node, result_enrichment_node,
    should_retry_after_verify,
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
        return "independent_signals"
    return END


def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("safety_check", safety_check_node)
    graph.add_node("independent_signals", independent_signals_node)
    graph.add_node("vision_detail", vision_detail_node)
    graph.add_node("clue_extract", clue_extract_node)
    graph.add_node("ocr_fusion", ocr_fusion_node)
    graph.add_node("anchor_search", anchor_search_node)
    graph.add_node("build_search_strategy", search_strategy_node)
    graph.add_node("react_loop", react_loop_node)
    graph.add_node("result_synthesize", result_synthesize_node)
    graph.add_node("adversarial_verify", adversarial_verify_node)
    graph.add_node("result_enrichment", result_enrichment_node)

    graph.set_entry_point("safety_check")

    graph.add_conditional_edges("safety_check", should_continue_after_safety, {
        "independent_signals": "independent_signals",
        END: END,
    })

    # Phase 1: independent signals, then blind visual clue extraction.
    graph.add_edge("independent_signals", "vision_detail")
    graph.add_edge("vision_detail", "clue_extract")
    graph.add_edge("clue_extract", "ocr_fusion")
    graph.add_edge("ocr_fusion", "anchor_search")

    # Phase 3: Strategy + ReAct loop
    graph.add_edge("anchor_search", "build_search_strategy")
    graph.add_edge("build_search_strategy", "react_loop")

    graph.add_conditional_edges("react_loop", should_continue_react, {
        "react_loop": "react_loop",
        "result_synthesize": "result_synthesize",
    })

    graph.add_edge("result_synthesize", "adversarial_verify")

    graph.add_conditional_edges("adversarial_verify", should_retry_after_verify, {
        "react_loop": "react_loop",
        END: "result_enrichment",
    })

    graph.add_edge("result_enrichment", END)

    return graph.compile()
