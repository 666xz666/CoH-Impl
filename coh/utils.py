from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams
import re

def remove_thinking(text):
    pattern = r"(?s)<think>.*?</think>"
    cleaned_text = re.sub(pattern, "", text)
    return cleaned_text

def parse_predict_answer(answer: str, dict) -> list:
    """
    parse LLM's predict to a list of entity name
    """
    answer = remove_thinking(answer)
    list = [x.split(".")[-1].strip().lower().replace("\\", "") for x in answer.strip().splitlines()[1:-1]]
    candidates = [x for x in list if x in dict.keys()]
    # print(candidates)
    return candidates

def parse_ids_to_list(output, list):
    """
    process LLM's output string
    "1, 2, 3, ..., 30"
    """
    output = remove_thinking(output)
    i_list = []
    for item in output.split(","):
        try:
            num = int(item.strip())
            list[num]
            i_list.append(num)
        except Exception:
            continue
    res = [list[i] for i in sorted(set(i_list))]
    # print(res)
    return res
    
def int_to_ordinal(num):
    """
    History Processing (CoH 3.1)
    48 -> "on the 2nd day"
    """
    
    if num < 0:
        raise ValueError("Input must be a non-negative integer")
    
    num = int(num / 24)
    
    # end with "11"、 "12"、 "13"
    if 10 <= num % 100 <= 13:
        suffix = "th"
    else:
        # "1"、 "2"、 "3"、 other
        last_digit = num % 10
        if last_digit == 1:
            suffix = "st"
        elif last_digit == 2:
            suffix = "nd"
        elif last_digit == 3:
            suffix = "rd"
        else:
            suffix = "th"
    
    return f"on the {num}{suffix} day"

  
def llm_generate(llm_instance, tokenizer_instance, generation_params, base_prompts: list[str]):
    """
    Generate outputs from the LLM based on the given prompts.

    Parameters:
    llm_instance: An instance of vllm.LLM or transformers.AutoModelForCausalLM.
    tokenizer_instance: The tokenizer associated with the LLM (required for transformers models).
    generation_params: A dictionary of generation parameters (e.g., temperature, top_p, max_tokens).
    base_prompts: A list of string prompts (unformatted, raw content).

    Returns:
    A list of generated strings.
    """
    processed_prompts = []

    # Check if it's a vLLM instance (assuming vllm.LLM is the type)
    # You might need to import LLM from vllm for this check to work
    try:
        from vllm import LLM as VLLMLLExample # Use an alias to avoid name conflict if LLM is also defined elsewhere
        is_vllm_instance = isinstance(llm_instance, VLLMLLExample)
    except ImportError:
        is_vllm_instance = False # vLLM not installed, assume transformers

    if is_vllm_instance:  # vllm.LLM
        # Get the tokenizer from the vLLM instance
        llm_tokenizer = llm_instance.get_tokenizer()

        # Apply chat template to each base prompt
        for prompt_content in base_prompts:
            messages = [{"role": "user", "content": prompt_content}]
            # Use apply_chat_template to format the prompt, important for Mixtral
            formatted_prompt = llm_tokenizer.apply_chat_template(
                messages,
                tokenize=False, # We want a string output
                add_generation_prompt=True # Add prompt to guide generation
            )
            processed_prompts.append(formatted_prompt)

        # Assuming SamplingParams needs to be imported from vllm
        try:
            from vllm import SamplingParams
            sampling_params = SamplingParams(**generation_params)
        except ImportError:
            raise ImportError("vLLM's SamplingParams not found. Please ensure vLLM is installed.")
        
        outputs = llm_instance.generate(processed_prompts, sampling_params=sampling_params)
        
        generated_texts = []
        for output in outputs:
            # vLLM's output.outputs[0].text is already decoded and skips special tokens by default
            generated_texts.append(output.outputs[0].text)
            
        # TODO: test line
        # print(processed_prompts[0])
        # print(generated_texts[0])
        
        return generated_texts
    else:  # transformers.AutoModelForCausalLM
        # Use the provided tokenizer for transformers models
        if tokenizer_instance is None:
            raise ValueError("Tokenizer must be provided for transformers models when using transformers models.")
        
        # Apply chat template to each base prompt
        for prompt_content in base_prompts:
            messages = [{"role": "user", "content": prompt_content}]
            # Use apply_chat_template to format the prompt for transformers
            formatted_prompt = tokenizer_instance.apply_chat_template(
                messages,
                tokenize=False, # We want a string output
                add_generation_prompt=True # Add prompt to guide generation
            )
            processed_prompts.append(formatted_prompt)

        # Tokenize the formatted prompts for transformers model
        inputs = tokenizer_instance(processed_prompts, return_tensors="pt", padding=True)
        inputs = {k: v.to(llm_instance.device) for k, v in inputs.items()}
        
        transformers_generate_params = {}
        if 'temperature' in generation_params:
            transformers_generate_params['temperature'] = generation_params['temperature']
        if 'top_p' in generation_params:
            transformers_generate_params['top_p'] = generation_params['top_p']
        if 'max_tokens' in generation_params:
            transformers_generate_params['max_new_tokens'] = generation_params['max_tokens']
        # Add other parameter mappings as needed, e.g., do_sample=True, num_beams=1
        transformers_generate_params['do_sample'] = generation_params.get('do_sample', True)
        transformers_generate_params['num_beams'] = generation_params.get('num_beams', 1)
        transformers_generate_params['pad_token_id'] = tokenizer_instance.eos_token_id # Often good practice

        outputs = llm_instance.generate(**inputs, **transformers_generate_params)
        # When decoding for transformers, explicitly set skip_special_tokens=True
        return tokenizer_instance.batch_decode(outputs, skip_special_tokens=True)