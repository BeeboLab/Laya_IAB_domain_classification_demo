import json
import time

from rl_agent_api import RLAgent


print("Loading Laya...")

t0 = time.perf_counter()

# "." = obecny folder, czyli:
# model.safetensors
# encoder/
# tokenizer/
# rl_agent_config.json
agent = RLAgent(".")

print(f"Loaded in {time.perf_counter() - t0:.2f}s")
print(f"Device: {agent.device}")


state = {
    "source_text": (
        "Taiwan Semiconductor Manufacturing Company, better known as TSMC, "
        "is participating in the development of Baipu Industrial Park in "
        "Kaohsiung, Taiwan. The park is intended to support advanced "
        "semiconductor packaging, equipment validation, materials testing "
        "and technical training. Baipu Industrial Park will cover "
        "approximately 88.7 hectares. The project remains under development."
    ),
    "candidate_entity": "Baipu Industrial Park"
}


questions = {

    "entity_type": {
        "type": "choice",
        "instructions": (
            "Which ontology type best describes candidate_entity "
            "according to source_text?"
        ),
        "criteria": {
            "INDUSTRIAL_PARK": "A geographically defined industrial or technology park.",
            "FACILITY": "A single physical building or operational site.",
            "COMPANY": "A commercial corporation.",
            "PROJECT": "An organized initiative or development project.",
            "CITY": "A city or municipality."
        }
    },

    "advanced_packaging": {
        "type": "noul",
        "instructions": (
            "Is candidate_entity materially associated with "
            "advanced semiconductor packaging in source_text?"
        )
    },

    "under_development": {
        "type": "noul",
        "instructions": (
            "Does source_text say candidate_entity is currently "
            "under development?"
        )
    },

    "primary_role": {
        "type": "choice",
        "instructions": (
            "What is the primary role of candidate_entity according "
            "to source_text?"
        ),
        "criteria": {
            "validation_and_packaging_support":
                "Validation, advanced packaging support, materials testing or training.",

            "high_volume_wafer_fabrication":
                "Primarily mass-production wafer fabrication.",

            "cloud_computing":
                "Cloud infrastructure or data-center services.",

            "residential_development":
                "Residential housing development."
        }
    }
}


print("\nRunning inference...")

t1 = time.perf_counter()

result = agent.system_one(
    state=state,
    questions=questions
)

elapsed = time.perf_counter() - t1

print(f"\nInference: {elapsed * 1000:.1f} ms\n")

print(json.dumps(result, indent=2, ensure_ascii=False))