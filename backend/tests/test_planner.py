from app.agent.planner import ReasoningPlanner


def test_planner_adds_youtube_transcript_before_summary():
    plan = ReasoningPlanner().create_plan("summarize", ["https://youtu.be/abc123xyz"])

    assert plan.index("youtube_transcript") < plan.index("summarize")


def test_planner_routes_sentiment():
    plan = ReasoningPlanner().create_plan("sentiment", [])

    assert "sentiment" in plan
    assert "synthesize_final_answer" == plan[-1]


def test_planner_routes_audio_pdf_to_comparison():
    plan = ReasoningPlanner().create_plan("qa", [], has_audio=True, has_pdf=True)

    assert "compare_content" in plan
    assert plan.index("compare_content") < plan.index("synthesize_final_answer")
