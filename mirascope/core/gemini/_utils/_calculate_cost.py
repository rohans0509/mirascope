"""Calculate the cost of a Gemini API call."""


def calculate_cost(
    input_tokens: int | float | None, output_tokens: int | float | None, model: str
) -> float | None:
    """Calculate the cost of a Gemini API call.
    
    https://ai.google.dev/pricing
    
    Model Variants:
    - gemini-2.0-flash-exp: Next generation features, speed, and multimodal generation
    - gemini-2.0-flash-lite: Lightweight version for faster responses
    - gemini-1.5-flash: Fast and versatile performance
    - gemini-1.5-pro: Complex reasoning tasks
    - gemini-1.5-flash-8b: High volume and lower intelligence tasks
    - gemini-1.0-pro: Natural language tasks (Deprecated on 2/15/2025)
    - text-embedding-004: Text embeddings
    
    Pricing (USD per 1M tokens):
    Model                Input               Output
    gemini-2.0-flash    $0.10               $0.40
    gemini-2.0-flash-lite $0.075            $0.30
    gemini-1.5-flash     $0.075              $0.30
    gemini-1.5-pro       $1.25               $5.00
    gemini-1.0-pro       $0.50               $1.50
    gemini-1.5-flash-8b  $0.0375             $0.15
    """
    # Base pricing for each model family
    base_pricing = {
        "gemini-2.0-flash": {
            "prompt": 0.000_000_1,
            "completion": 0.000_000_4,
        },
        "gemini-2.0-flash-lite": {
            "prompt": 0.000_000_075,
            "completion": 0.000_000_3,
        },
        "gemini-1.5-flash": {
            "prompt": 0.000_000_075,
            "completion": 0.000_000_3,
        },
        "gemini-1.5-pro": {
            "prompt": 0.000_001_25,
            "completion": 0.000_005,
        },
        "gemini-1.0-pro": {
            "prompt": 0.000_000_5,
            "completion": 0.000_001_5,
        },
        "gemini-1.5-flash-8b": {
            "prompt": 0.000_000_0375,
            "completion": 0.000_000_15,
        },
    }

    if input_tokens is None or output_tokens is None:
        return None

    # Extract the base model from the version-specific name
    # This handles patterns like: gemini-1.5-flash-latest, gemini-1.5-flash-001, etc.
    base_model = "-".join(model.split("-")[:3])
    
    try:
        model_pricing = base_pricing[base_model]
    except KeyError:
        return None

    prompt_cost = input_tokens * model_pricing["prompt"]
    completion_cost = output_tokens * model_pricing["completion"]
    total_cost = prompt_cost + completion_cost

    return total_cost
