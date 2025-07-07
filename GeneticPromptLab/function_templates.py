function_templates = [
    {
        "name": "generate_prompts",
        "description": "You are a Prompt-Engineering assistant whose objective is to understand the given task and come up with an effective prompt which enables the model to answer questions accurately.\n\nYour task is to create a generalized prompt that works across all classes/categories. The prompt should be inspired by the given examples but should NOT simply repeat question-answer pairs. Instead, provide specific guidance that helps an LLM classify correctly when simple generic prompts fail.\n\nIMPORTANT: You are part of a genetic algorithm, so introduce some variation and specificity based on the examples provided.\n\nFormat your response as:\nPROMPT: [your generated prompt here]",
        "parameters": {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "This is a prompt which if given to an LLM, will enable it to correctly classify the given questions for the specified labels."
                }
            },
            "required": ["prompt"]
        }
    },
    {
        "name": "QnA_bot",
        "description": "You are a Question-Answering classification bot. Your task is to choose the correct label for each question based on the given instructions and available options.\n\nIMPORTANT: You must provide exactly the number of labels requested, in the same order as the questions.\n\nFormat your response as:\nLABELS: [label1, label2, label3, ...]",
        "parameters": {
            "type": "object",
            "properties": {
                "label_array": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "label": {
                                "description": "Most correct label based on the given queries and instructions in the same order provided. Labels allowed: ",
                                "type": "string",
                                "enum": []
                            },
                        },
                        "required": ["label"],
                    },
                    "description": "An array of labels corresponding to each question in order.",
                    "minItems": 0,
                    "maxItems": 100,
                }
            },
            "required": ["label_array"]
        }
    },
    {
        "name": "prompt_mutate",
        "description": "You are part of a Genetic-Optimization Algorithm. Your objective is to mutate a prompt to introduce controlled randomness while maintaining the core task functionality.\n\nThe mutation should:\n1. Rephrase and modify the original prompt\n2. Keep the essential task requirements\n3. Add new perspectives or approaches\n4. Maintain effectiveness for the classification task\n\nFormat your response as:\nMUTATED_PROMPT: [your mutated prompt here]",
        "parameters": {
            "type": "object",
            "properties": {
                "mutated_prompt": {
                    "type": "string",
                    "description": "A mutated version of the original prompt that maintains core functionality while introducing beneficial variations. Mutation intensity: "
                },
            },
            "required": ["mutated_prompt"]
        }
    },
    {
        "name": "prompt_crossover",
        "description": "You are part of a Genetic-Optimization Algorithm. Your objective is to create a child prompt by combining two parent prompts: a Template (control) prompt and an Additive prompt.\n\nYour task:\n1. Use the Template prompt as the base structure\n2. Incorporate beneficial elements from the Additive prompt\n3. Create a more detailed and effective prompt\n4. Add stronger directions and more complex instructions\n5. Learn from the parent prompts' performance to improve accuracy\n\nThe child prompt should be more sophisticated than either parent and address classification weaknesses observed in the examples.\n\nFormat your response as:\nCHILD_PROMPT: [your child prompt here]",
        "parameters": {
            "type": "object",
            "properties": {
                "child_prompt": {
                    "type": "string",
                    "description": "A child prompt created by combining the template/control prompt with beneficial elements from the additive prompt. Must be different from both parents and more effective."
                },
            },
            "required": ["child_prompt"]
        }
    }
]