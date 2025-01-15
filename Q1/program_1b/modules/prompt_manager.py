def create_prompt(case):
    """
    Generate a tailored prompt based on the use case.
    """
    prompts = {
        "claims": "You are a helpful assistant for medical insurance claims.",
        "coverage": "You are a knowledgeable assistant explaining medical coverage details."
    }
    return prompts.get(case, "You are a friendly and helpful medical insurance assistant.")
