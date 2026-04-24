"""
GLAM Project Idea Validator
Uses Claude Opus 4.6 with adaptive thinking to deeply analyze the project feasibility,
risks, datasets, approaches, and provide a comprehensive research roadmap.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
import anthropic

load_dotenv()

PROJECT_DESCRIPTION = """
Project: GLAM - Predicting the progression rate in highly myopic glaucomatous patients

Goal:
- Predict the rate of progression in dB/year of MD (mean deviation)
- Predict %/year of VFI (visual field index)
- Subjects: highly myopic AND glaucomatous patients
- Approach: Deep Learning

Required skills: Deep Learning, Data Management

Context:
- Myopic glaucoma is a particularly challenging condition where both high myopia (≥6D or axial
  length ≥26mm) and glaucoma co-exist
- Standard glaucoma progression criteria may not apply due to myopia-related structural changes
- Visual field (VF) tests are the gold standard for functional progression
- MD (Mean Deviation) measures overall field loss in dB; VFI (Visual Field Index) is a global
  percentage index of VF health
- Predicting annual rates of change is crucial for clinical management decisions
"""

ANALYSIS_PROMPT = f"""
You are a world-class expert in ophthalmology AI research, medical deep learning, and clinical
data science. Analyze this research project proposal with rigor and depth.

<project>
{PROJECT_DESCRIPTION}
</project>

Provide a comprehensive analysis covering ALL of the following sections:

## 1. IDEA VALIDATION
- Scientific novelty and significance (why this matters clinically)
- Feasibility assessment (can this be done with current DL technology?)
- Unique challenges of myopic glaucoma vs regular glaucoma progression prediction
- What makes this harder than standard glaucoma progression prediction

## 2. TECHNICAL APPROACH RECOMMENDATIONS
- Best deep learning architectures for this task (be specific: ConvNeXt-V2, Hybrid-VF-Net, etc.)
- Multimodal fusion strategy (OCT + Visual Field + Clinical data)
- Temporal modeling approach (longitudinal data handling)
- Loss functions for regression of dB/year and %/year
- How to handle the dual output (MD and VFI simultaneously)

## 3. DATA REQUIREMENTS & DATASETS
- What data modalities are essential vs optional
- Public datasets available (GRAPE, UWHVF, Harvard Glaucoma, etc.)
- Sample size requirements for reliable training
- Key features to extract from visual field reports
- Data augmentation strategies for medical time-series

## 4. CHALLENGES & RISKS
- Clinical challenges (myopia-glaucoma interaction, variable follow-up)
- Technical challenges (class imbalance, missing data, variable visit intervals)
- How to mitigate each risk

## 5. EVALUATION STRATEGY
- Metrics for regression tasks (MAE, RMSE in dB/year and %/year)
- Clinical validity metrics (progression detection time vs clinician)
- Cross-validation strategy for temporal medical data
- What constitutes a clinically meaningful result

## 6. RESEARCH ROADMAP (6-12 months)
- Phase 1: Data pipeline and baseline models
- Phase 2: Advanced multimodal architecture
- Phase 3: Clinical validation and interpretability
- Key milestones and deliverables

## 7. VERDICT
- Overall feasibility score (1-10) with justification
- Top 3 recommendations to maximize success
- One critical risk that could derail the project and how to avoid it

Be direct, specific, and assume the reader is a deep learning researcher (not a clinician).
"""


def stream_analysis(client: anthropic.Anthropic) -> str:
    """Stream the analysis from Claude with adaptive thinking."""
    print("\n" + "="*70)
    print("  GLAM PROJECT — IDEA VALIDATION via Claude Opus 4.6")
    print("="*70)
    print("\nAnalyzing project feasibility with extended reasoning...\n")
    print("-"*70)

    full_response = []
    thinking_shown = False

    with client.messages.stream(
        model="claude-opus-4-6",
        max_tokens=8000,
        thinking={"type": "adaptive"},
        messages=[{"role": "user", "content": ANALYSIS_PROMPT}],
    ) as stream:
        for event in stream:
            if event.type == "content_block_start":
                if event.content_block.type == "thinking" and not thinking_shown:
                    print("[Claude is reasoning deeply about your project...]\n")
                    thinking_shown = True
            elif event.type == "content_block_delta":
                if event.delta.type == "text_delta":
                    print(event.delta.text, end="", flush=True)
                    full_response.append(event.delta.text)

        final = stream.get_final_message()

    print("\n" + "-"*70)
    print(f"\nTokens used — Input: {final.usage.input_tokens:,} | Output: {final.usage.output_tokens:,}")
    return "".join(full_response)


def save_report(analysis: str) -> Path:
    """Save the analysis report to a markdown file."""
    output_dir = Path("reports")
    output_dir.mkdir(exist_ok=True)
    report_path = output_dir / "idea_validation_report.md"

    report_content = f"""# GLAM Project — Idea Validation Report

> Generated by Claude Opus 4.6 with Adaptive Thinking

---

{analysis}

---
*Report generated via Claude API · Model: claude-opus-4-6 · Thinking: adaptive*
"""
    report_path.write_text(report_content)
    print(f"\nReport saved to: {report_path}")
    return report_path


def main() -> None:
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY not set.")
        print("  1. Copy .env.example to .env")
        print("  2. Add your Anthropic API key")
        sys.exit(1)

    client = anthropic.Anthropic(api_key=api_key)

    try:
        analysis = stream_analysis(client)
        save_report(analysis)
        print("\nIdea validation complete!")
    except anthropic.AuthenticationError:
        print("ERROR: Invalid API key. Check your ANTHROPIC_API_KEY.")
        sys.exit(1)
    except anthropic.RateLimitError:
        print("ERROR: Rate limited. Please wait and try again.")
        sys.exit(1)
    except anthropic.APIError as e:
        print(f"ERROR: API error — {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
