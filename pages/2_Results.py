import streamlit as st
import yaml
import re
from datetime import datetime
from pathlib import Path

st.markdown(
    """
<style>
    .stApp > div:first-child > div:first-child > div:first-child {
        max-width: 100% !important;
    }
    section.main {
        max-width: 100% !important;
    }
    .stExpander {
        max-width: 100% !important;
    }
    .stMarkdown {
        max-width: 100% !important;
    }
    .stImage {
        max-width: 100% !important;
    }
    div[data-testid="stHorizontalBlock"] {
        max-width: 100% !important;
    }
</style>
""",
    unsafe_allow_html=True,
)

RESULTS_FILE = Path(__file__).parent.parent.resolve() / "experiments" / "results.yaml"
MARKDOWN_RESULT_FILE = Path(__file__).parent.parent.resolve() / "results.md"
EXPERIMENTS_DIR = Path(__file__).parent.parent.resolve() / "experiments"


def load_results():
    if not RESULTS_FILE.exists():
        return []

    with open(RESULTS_FILE, "r") as f:
        data = yaml.safe_load(f)

    if data is None or "results" not in data:
        return []

    return data.get("results", [])


def load_markdown_results():
    if not MARKDOWN_RESULT_FILE.exists():
        return []

    with open(MARKDOWN_RESULT_FILE, "r") as f:
        content = f.read()

    pattern = r"#\s*Results:\s*(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2})\s*\(([^)]+)\)\s*\n*(.*?)(?=\n---|\n#\s*Results:|$)"

    matches = re.findall(pattern, content, re.DOTALL)

    results = []
    for match in matches:
        timestamp = match[0].strip()
        title = match[1].strip()
        markdown_content = match[2].strip()
        results.append(
            {"timestamp": timestamp, "title": title, "content": markdown_content}
        )

    results.sort(key=lambda x: x["timestamp"], reverse=True)

    return results


def get_section_image(section_key):
    image_map = {
        "neuroanatomical": "subject_episode+parcel_trials_clustering/UMAP@n_neighbors=50.png",
        "temporal": "subject_episode+parcel+window_subsample_trials/UMAP@n_neighbors=50.png",
        "scaling": [
            "subject_episode+parcel_trials_clustering/UMAP@n_neighbors=50.png",
            "subject_episode+parcel_trials_clustering/v1/UMAP@n_neighbors=50.png",
        ],
        "robustness": "subject+multi_episodes+parcel_trials_clustering/UMAP@n_neighbors=50.png",
        "hcptrt": "hcptrt_across_tasks/UMAP_all_tasks@n_neighbors=15.png",
        "fingerprinting": "hcptrt_across_tasks/UMAP_all_tasks@n_neighbors=15.png",
    }
    return image_map.get(section_key, [])


def render_status_dashboard(timestamp):
    st.markdown(
        f"""
    <div style="background: linear-gradient(90deg, #1e3a5f 0%, #2d5a87 100%); 
                padding: 20px; border-radius: 10px; margin-bottom: 20px;">
        <h2 style="color: white; margin-bottom: 5px;">📊 TRACE Pipeline Validation Results</h2>
        <p style="color: #a0c4ff; margin-bottom: 15px;">Last Updated: {timestamp}</p>
        <div style="display: flex; gap: 15px; flex-wrap: wrap;">
            <span style="background: #28a745; color: white; padding: 8px 16px; 
                         border-radius: 20px; font-size: 14px;">✅ HCPTRT Validated</span>
            <span style="background: #17a2b8; color: white; padding: 8px 16px; 
                         border-radius: 20px; font-size: 14px;">🚀 Next: TRACE Training</span>
            <span style="background: #ffc107; color: #333; padding: 8px 16px; 
                         border-radius: 20px; font-size: 14px;">🧠 MMP (360 Parcels)</span>
            <span style="background: #6c757d; color: white; padding: 8px 16px; 
                         border-radius: 20px; font-size: 14px;">🌍 CIFTI Grayordinates</span>
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("HCPTRT Cross-Task", "✅ Validated", "6 Domains + Rest")
    with col2:
        st.metric("Algonauts Temporal", "✅ Validated", "Stable across windows")
    with col3:
        st.metric("Anatomical Resolution", "🚀 High", "MMP Surface Mesh")
    with col4:
        st.metric("Denoising", "✅ Complete", "24 Motion + CSF/WM")


def render_finding_section(title, content, image_path=None, caption=None):
    with st.expander(f"▶ {title}", expanded=True):
        if image_path:
            full_path = EXPERIMENTS_DIR / image_path
            if full_path.exists():
                st.image(
                    str(full_path), caption=caption or title, use_container_width=True
                )
            else:
                st.warning(f"Image not found: {image_path}")

        st.markdown(content)

        if "cluster" in content.lower() or "anatomical" in title.lower():
            st.success("✓ Key Finding: Data preserves known brain architecture")
        elif "temporal" in content.lower() or "stability" in title.lower():
            st.info("✓ Key Finding: Validates positive pair sampling strategy")


def render_summary_with_images(markdown_content):
    sections = []
    current_section = None
    current_content = []

    lines = markdown_content.split("\n")
    for line in lines:
        if line.startswith("## "):
            if current_section:
                sections.append(
                    {"title": current_section, "content": "\n".join(current_content)}
                )
            current_section = line.replace("## ", "").strip()
            current_content = []
        elif line.startswith("### "):
            if current_section:
                sections.append(
                    {"title": current_section, "content": "\n".join(current_content)}
                )
            current_section = line.replace("### ", "").strip()
            current_content = []
        else:
            current_content.append(line)

    if current_section:
        sections.append(
            {"title": current_section, "content": "\n".join(current_content)}
        )

    for section in sections:
        title = section["title"]
        content = section["content"]

        title_lower = title.lower()

        if "summary" in title_lower:
            st.markdown("## " + title)
            st.markdown(content)

        elif "cross-task" in title_lower and "stability" in title_lower:
            render_finding_section(
                "1. Cross-Task Anatomical Stability",
                content,
                "hcptrt_across_tasks/UMAP_all_tasks@n_neighbors=15.png",
                "UMAP embedding of 360 MMP regions across 6 cognitive tasks + Rest",
            )

        elif "fingerprinting" in title_lower:
            render_finding_section(
                "2. Multi-Task Fingerprinting",
                content,
                "hcptrt_across_tasks/UMAP_all_tasks@n_neighbors=15.png",
                "Regions cluster by anatomical identity regardless of the current task",
            )

        elif "space distinction" in title_lower or "grayordinate" in title_lower:
            st.markdown("## 🌍 Space Distinction")
            st.markdown(content)
            st.info(
                "**Key Distinction:** CIFTI/Surface measuring prevents signal bleeding across folds compared to Volumetric/MNI."
            )

        elif "neuroanatomical" in title_lower:
            render_finding_section(
                "1. Neuroanatomical Consistency",
                content,
                "subject_episode+parcel_trials_clustering/UMAP@n_neighbors=50.png",
                "UMAP embedding colored by Lobe (Left) and Hemisphere (Right)",
            )

        elif "temporal" in title_lower and "stability" in title_lower:
            render_finding_section(
                "2. Temporal Stability",
                content,
                "subject_episode+parcel+window_subsample_trials/UMAP@n_neighbors=50.png",
                "Parcel identity preserved across windowed subsamples",
            )

        elif "scaling" in title_lower:
            col1, col2 = st.columns(2)
            with col1:
                render_finding_section(
                    "3a. Local Scaling (minmax_time)",
                    "Normalizes each parcel's activity relative to its own range.\nEmphasizes relative temporal dynamics.",
                    "subject_episode+parcel_trials_clustering/UMAP@n_neighbors=50.png",
                    "Local/Time Scaling",
                )
            with col2:
                render_finding_section(
                    "3b. Global Scaling (minmax)",
                    "Normalizes activity across entire dataset.\nPreserves absolute magnitude differences between parcels.",
                    "subject_episode+parcel_trials_clustering/v1/UMAP@n_neighbors=50.png",
                    "Global Scaling (v1)",
                )
            st.info(
                "**Key Insight:** Global scaling tightens anatomical clusters; Local scaling emphasizes temporal dynamics."
            )

        elif "presentation" in title_lower or "strategy" in title_lower:
            st.markdown("## 📽️ Presentation Strategy")
            st.markdown(content)

            st.markdown("### Key Visualization Slides")

            if "HCPTRT" in content or "Cross-Task" in content:
                slides = [
                    (
                        "Slide 1: Universal Functional Identity",
                        "hcptrt_across_tasks/UMAP_all_tasks@n_neighbors=15.png",
                        "MMP regions across 6 cognitive tasks + Rest",
                    ),
                    (
                        "Slide 2: Embedding Robustness",
                        "hcptrt_across_tasks/tSNE_all_tasks@perplexity=15.png",
                        "t-SNE vs UMAP consistency check",
                    ),
                    (
                        "Slide 3: Surface Advantage",
                        "hcptrt_across_tasks/UMAP_all_tasks@n_neighbors=15.png",
                        "Resolution comparison (Surface vs Volume)",
                    ),
                ]
            else:
                slides = [
                    (
                        "Slide 1: Anatomical Organization",
                        "subject_episode+parcel_trials_clustering/UMAP@n_neighbors=50.png",
                        "Single episode, Subject 1 - clustering by Lobe and Hemisphere",
                    ),
                    (
                        "Slide 2: Scaling Sensitivity",
                        "subject_episode+parcel_trials_clustering/v1/UMAP@n_neighbors=50.png",
                        "Global vs Local normalization comparison",
                    ),
                    (
                        "Slide 3: Temporal Persistence",
                        "subject_episode+parcel+window_subsample_trials/UMAP@n_neighbors=50.png",
                        "Stable fingerprint across movie episode",
                    ),
                    (
                        "Slide 4: Robustness Across Stimuli",
                        "subject+multi_episodes+parcel_trials_clustering/UMAP@n_neighbors=50.png",
                        "10 episodes - functional-anatomical organization robust",
                    ),
                ]

            for i, (slide_title, img_path, desc) in enumerate(slides, 1):
                with st.expander(f"Slide {i}: {slide_title}"):
                    full_path = EXPERIMENTS_DIR / img_path
                    if full_path.exists():
                        st.image(str(full_path), caption=desc, use_container_width=True)
                    else:
                        st.warning(f"Image not found: {img_path}")
                    st.caption(f"**Talking Point:** {desc}")

            with st.expander("Slide 5: Path Forward"):
                st.markdown("""
                **Conclusion:** The pipeline is validated.
                
                **Next Steps:** Proceed to train the Contrastive Embedding model 
                using these validated positive/negative sampling strategies.
                """)
                st.success("🚀 Ready for TRACE Embedding Training")

        elif "limitation" in title_lower:
            st.markdown("## ⚠️ Current Limitations")
            st.markdown(content)
            st.warning("""
            **Impact on TRACE:**
            - Single trial data limits true stimulus-repetition pairs
            - Using temporal subsampling as workaround
            - Scaling method still being evaluated
            """)

        elif "next step" in title_lower:
            st.markdown("## 🚀 Next Steps")
            st.markdown(content)
            st.info("""
            **Immediate Priority:**
            1. Incorporate HCP data (15 trials per subject-scene)
            2. Finalize scaling method selection
            3. Initialize TRACE contrastive embedding training
            """)


st.title("Results")

st.markdown("---")

markdown_results = load_markdown_results()

if markdown_results:
    for i, markdown_result in enumerate(markdown_results):
        expanded = i == 0
        with st.expander(
            f"📊 {markdown_result['timestamp']} - {markdown_result['title']}",
            expanded=expanded,
        ):
            if any(k in markdown_result["title"] for k in ["Pipeline Validation", "HCPTRT", "Cross-Task"]):
                render_status_dashboard(markdown_result["timestamp"])
                st.markdown("---")
                st.markdown("### Detailed Findings")
                render_summary_with_images(markdown_result["content"])
            else:
                st.markdown(markdown_result["content"])

results = load_results()

st.markdown("---")
st.markdown("## 📁 Experiment Results (YAML)")

if not results:
    st.info("No experiment results yet - check back later as experiments progress.")

    st.markdown("""
    To add a result, edit `experiments/results.yaml`:

    ```yaml
    results:
      - date: "2025-04-13"
        title: "Your result title"
        description: "Description of what was done"
        metric: 0.85
    ```
    """)
else:
    results_sorted = sorted(results, key=lambda x: x.get("date", ""), reverse=True)

    st.markdown(f"**{len(results_sorted)} result(s) recorded**")

    for i, res in enumerate(results_sorted):
        with st.expander(
            f"{res.get('date', 'Unknown date')} - {res.get('title', 'Untitled')}",
            expanded=False,
        ):
            st.markdown(f"**Date:** {res.get('date', 'N/A')}")

            st.markdown(f"**Title:** {res.get('title', 'N/A')}")

            st.markdown(f"**Description:** {res.get('description', 'No description')}")

            metric = res.get("metric")
            if metric is not None:
                st.metric("Metric Value", metric)

st.markdown("---")

st.caption(f"Results loaded from: {RESULTS_FILE}")

st.sidebar.markdown("""
**Results**

Track experiment outcomes over time.

[Home](../Home)
[Progress](../Progress)
""")

st.sidebar.page_link("app.py", label="← Back to Home")
