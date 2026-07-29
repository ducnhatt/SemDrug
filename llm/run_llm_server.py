import argparse
import logging
import json
import os
import re
from pathlib import Path

from typing import Union
import pandas as pd
from datasets import Dataset
from vllm import LLM, SamplingParams
from transformers import set_seed

try:
    from huggingface_hub import login
except Exception:
    login = None

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass


def maybe_login_huggingface() -> None:
    """Login only when HF_API_KEY is provided.

    On Vast/servers, you may also run `huggingface-cli login` once.
    This function avoids failing when `.env` is absent or when using a local model path.
    """
    token = os.getenv("HF_API_KEY") or os.getenv("HUGGING_FACE_HUB_TOKEN") or os.getenv("HF_TOKEN")
    if token and login is not None:
        login(token=token)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# DRUGBANK_WITH_OPTIONS_PROMPT = (
#     "You are a pharmacodynamics expert. Answer the questions using the given knowledge graph information (if available) and your medical expertise."
#     "Base your answer on evidence of known interaction mechanisms, pharmacological effects, or similarities to related compounds if available"
#     "Select the option number that best describes the interaction type."
#     "Your answer must be concise and formatted as follows: ##Answer: <DrugX the specific effect or mechanism of interaction DrugY>.\n"
# )

# DRUGBANK_WITH_OPTIONS_PROMPT = (
#     "You are performing a drug-drug interaction mechanism classification task.\n"
#     "Choose the option that best describes the specific interaction effect or mechanism "
#     "between Drug1 and Drug2.\n\n"
#     "Use the given knowledge graph information and drug descriptions when available.\n"
#     "Focus on the primary interaction mechanism, such as metabolism, serum concentration, "
#     "absorption, bioavailability, excretion, therapeutic efficacy, adverse effects, QTc prolongation, "
#     "bleeding risk, CNS depression, cardiotoxicity, nephrotoxicity, hepatotoxicity, or related pharmacological effects.\n"
#     "Your answer must exactly match one of the provided options.\n"
#     "Format exactly as: ##Answer: <Drug1 specific effect or mechanism Drug2>.\n"
# )

DRUGBANK_WITH_OPTIONS_PROMPT = (
    "You are performing a drug-drug interaction mechanism classification task.\n"
    "Choose the option that best describes the specific interaction effect or mechanism "
    "between Drug1 and Drug2.\n\n"
    "Use the given knowledge graph information when available.\n"
    "Focus on evidence of known interaction mechanisms, pharmacological effects, or similarities to related compounds if available.\n"
    "Your answer must contain exactly one label from the OPTIONS list."
    "If multiple options seem plausible, choose the single closest matching option from the OPTIONS list.\n"
    "Format exactly as: ##Answer: <Drug1 specific effect or mechanism Drug2>.\n"
)


#open drugbank KG prompt drugbank
DRUGBANK_OPEN_PROMPT = (
    "You are a pharmacodynamics expert. Answer the questions using the given knowledge graph information (if available), essential parts of the drug definitions, and your medical expertise.\n"  
    "Base your answer on evidence of known interaction mechanisms, pharmacological effects, or similarities to related compounds if applicable.\n"
    "Avoid generalizations unless directly supported by the knowledge graph.\n"
    "Focus exclusively on the primary interaction type between the specified drugs.Exclude speculative details or unrelated pathways unless they directly contribute to the stated interaction.\n"
    "Your answer must be concise and formatted as follows: ##Answer: <DrugX the specific effect or mechanism of interaction DrugY>.\n"
)


## KG prompt ddinter
# DDINTER_PROMPT = (
#     "You are a pharmacodynamics expert. Answer the questions using the given knowledge graph information (if available), essential parts of the drug definitions, and your medical expertise.\n" 
#     "Base your answer on evidence of known interaction mechanisms, pharmacological effects, or similarities to related compounds if applicable.\n"
#     "Avoid generalizations unless directly supported by the knowledge graph.\n"
#     "Focus exclusively on the primary interaction type between the specified drugs. Exclude speculative details or unrelated pathways unless they directly contribute to the stated interaction.\n"
#     "Your answer must be concise and formatted as follows: ##Answer: <the specific level of severity>.\n"
# )



DDINTER_PROMPT = (
    "You are performing a three-class drug-drug interaction severity classification task.\n"
    "Choose exactly one label from: Major, Moderate, Minor.\n\n"

    "Decision rules:\n"
    "- Choose Major when the interaction may cause severe adverse effects, life-threatening toxicity, dangerous changes in serum concentration, strong QT prolongation, severe CNS/respiratory depression, severe bleeding, severe hepatotoxicity, nephrotoxicity, cardiotoxicity, or requires medical intervention.\n"
    "- Choose Moderate when the interaction may worsen the patient's condition, require monitoring, dosage adjustment, or a change in therapy, but is not clearly severe or life-threatening.\n"
    "- Choose Minor only when the interaction has limited clinical effect and usually does not require therapy change.\n"
    "- Do not choose Minor merely because the mechanism is unclear.\n\n"

    "Use the given knowledge graph information and drug descriptions when available.\n"
    "Output exactly one label.\n"
    "Format exactly as: ##Answer: <Major|Moderate|Minor>."
)

PHARMACOTHERAPYDB_KG_PROMPT = (
    "You are a pharmacodynamics expert. Answer the questions using the given knowledge graph information (if available), essential parts of the drug definitions, and your medical expertise.\n"
    "Base your answer on evidence of known interaction mechanisms, pharmacological effects, or similarities to related compounds if applicable.\n"
    "Avoid generalizations unless directly supported by the knowledge graph.\n"
    "Focus exclusively on the therapeutic indication between the specified drugs and conditions. Exclude speculative details or unrelated pathways unless they directly support the indication.\n"
    "Your answer must be concise and formatted exactly as follows: ##Answer: <Indications>.\n"
)


# Base-model-only prompt (used when KG is not included)
PHARMACOTHERAPYDB_BASE_PROMPT = (
    "You are a pharmacodynamics expert. Answer the questions using your medical expertise.\n"
    "Base your answer on evidence of known interaction mechanisms, pharmacological effects, or similarities to related compounds if applicable.\n"
    "Focus exclusively on the therapeutic indication between the specified drugs and conditions.\n"
    "Your answer must be concise and formatted exactly as follows: ##Answer: <Indications>.\n"
)


SYSTEM_INSTRUCTION_PROMPT = {
    "drugbank_with_options": DRUGBANK_WITH_OPTIONS_PROMPT,
    "drugbank_with_bullets": DRUGBANK_WITH_OPTIONS_PROMPT,
    "drugbank_open": DRUGBANK_OPEN_PROMPT,
    "ddinter_common": DDINTER_PROMPT,
    "pharmacotherapydb": lambda use_kg: PHARMACOTHERAPYDB_KG_PROMPT if use_kg else PHARMACOTHERAPYDB_BASE_PROMPT,
}

DRUGBANK_OPTIONS = {
    0: "{u} may increase the photosensitizing activities of {v}",
    1: "{u} may increase the anticholinergic activities of {v}",
    2: "{u} can decrease the bioavailability of {v}",
    3: "{u} can increase the metabolism of {v}",
    4: "{u} may decrease the vasoconstricting activities of {v}",
    5: "{u} may increase the anticoagulant activities of {v}",
    6: "{u} may increase the ototoxic activities of {v}",
    7: "{u} can increase the therapeutic efficacy of {v}",
    8: "{u} may increase the hypoglycemic activities of {v}",
    9: "{u} may increase the antihypertensive activities of {v}",
    10: "{u} may reduce the serum concentration of the active metabolites of {v}",
    11: "{u} may decrease the anticoagulant activities of {v}",
    12: "{u} may decrease the absorption of {v}",
    13: "{u} may decrease the bronchodilatory activities of {v}",
    14: "{u} may increase the cardiotoxic activities of {v}",
    15: "{u} may increase the central nervous system depressant activities of {v}",
    16: "{u} may decrease the neuromuscular blocking activities of {v}",
    17: "{u} can increase the absorption and serum concentration of {v}",
    18: "{u} may increase the vasoconstricting activities of {v}",
    19: "{u} may increase the QTc prolonging activities of {v}",
    20: "{u} may increase the neuromuscular blocking activities of {v}",
    21: "{u} may increase the adverse neuromuscular activities of {v}",
    22: "{u} may increase the stimulatory activities of {v}",
    23: "{u} may increase the hypocalcemic activities of {v}",
    24: "{u} may increase the atrioventricular blocking activities of {v}",
    25: "{u} may decrease the antiplatelet activities of {v}",
    26: "{u} may increase the neuroexcitatory activities of {v}",
    27: "{u} may increase the dermatologic adverse activities of {v}",
    28: "{u} may decrease the diuretic activities of {v}",
    29: "{u} may increase the orthostatic hypotensive activities of {v}",
    30: "{u} may increase the hypertensive effects of {v}",
    31: "{u} may increase the sedative activities of {v}",
    32: "{u} may increase the severity of QTc prolonging effects when combined with {v}",
    33: "{u} may increase the immunosuppressive activities of {v}",
    34: "{u} may increase the neurotoxic activities of {v}",
    35: "{u} may increase the antipsychotic activities of {v}",
    36: "{u} may decrease the antihypertensive activities of {v}",
    37: "{u} may increase the vasodilatory activities of {v}",
    38: "{u} may increase the constipating activities of {v}",
    39: "{u} may increase the respiratory depressant activities of {v}",
    40: "{u} may increase the hypotensive and central nervous system depressant activities of {v}",
    41: "{u} may increase the severity of hyperkalemic effects when combined with {v}",
    42: "{u} may decrease the protein binding of {v}",
    43: "{u} may increase the central neurotoxic activities of {v}",
    44: "{u} may decrease the diagnostic effectiveness of {v} ",
    45: "{u} may increase the bronchoconstrictory activities of {v}",
    46: "{u} may decrease the metabolism of {v}",
    47: "{u} may increase the myopathic rhabdomyolysis activities of {v}",
    48: "{u} may increase the severity of adverse effects when combined with {v}",
    49: "{u} may increase heart failure risks when combined with {v}",
    50: "{u} may increase the hypercalcemic activities of {v}",
    51: "{u} may decrease the analgesic activities of {v}",
    52: "{u} may increase the antiplatelet activities of {v}",
    53: "{u} may increase the bradycardic activities of {v}",
    54: "{u} may increase the hyponatremic activities of {v}",
    55: "{u} may increase the hypotensive effects when combined with {v}",
    56: "{u} may increase the nephrotoxic activities of {v}",
    57: "{u} may decrease the cardiotoxic activities of {v}",
    58: "{u} may increase the ulcerogenic activities of {v}",
    59: "{u} may increase the hypotensive activities of {v}",
    60: "{u} may decrease the stimulatory activities of {v}",
    61: "{u} may increase the bioavailability of {v}",
    62: "{u} may increase the myelosuppressive activities of {v}",
    63: "{u} may increase the serotonergic activities of {v}",
    64: "{u} may increase the excretion rate {v}",
    65: "{u} may increase bleeding risks when combined with {v}",
    66: "{u} may decrease the absorption and serum concentration of {v}",
    67: "{u} may increase the hyperkalemic activities of {v}",
    68: "{u} may increase the analgesic activities of {v}",
    69: "{u} may decrease the therapeutic efficacy of {v}",
    70: "{u} may increase the hypertensive activities of {v}",
    71: "{u} may decrease the excretion rate  {v}",
    72: "{u} may increase the serum concentration of {v}",
    73: "{u} may increase the fluid retaining activities of {v}",
    74: "{u} may decrease the serum concentration of {v}",
    75: "{u} may decrease the sedative activities of {v}",
    76: "{u} may increase the serum concentration of the active metabolites of {v}",
    77: "{u} may increase the hyperglycemic activities of {v}",
    78: "{u} may increase the central nervous system depressant and hypertensive activities of {v}",
    79: "{u} may increase the hepatotoxic activities of {v}",
    80: "{u} may increase the thrombogenic activities of {v}",
    81: "{u} may increase the arrhythmogenic activities of {v}",
    82: "{u} may increase the hypokalemic activities of {v}",
    83: "{u} may increase the vasopressor activities of {v}",
    84: "{u} may increase the tachycardic activities of {v}",
    85: "{u} may increase the risk of hypersensitivity reaction to {v}",
}


DDINTER_OPTIONS = {
    "Major": "The interactions are life-threatening and/or require medical treatment or intervention to minimize or prevent severe adverse effects.",
    "Moderate": "The interactions may result in exacerbation of the disease of the patient and/or change in therapy.",
    "Minor": "The interactions would limit the clinical effects. The manifestations may include an increase in frequency or severity of adverse effects, but usually they do not require changes in therapy.",
}


PHARMACOTHERAPYDB_OPTIONS = {
    "Disease-modifying": "The drug therapeutically changes the underlying or downstream biology of the disease.",
    "Palliates": "The drug only alleviates symptoms without altering disease progression.",
    "Non-indication": "The drug neither therapeutically changes the disease nor treats its symptoms.",
}

# PHARMACOTHERAPYDB_OPTIONS = {
#     "Disease-modifying": "The drug therapeutically treats, prevents, slows, reverses, or modifies the underlying disease process or progression.",
#     "Palliates": "The drug only relieves symptoms or provides supportive care without modifying the underlying disease.",
#     "Non-indication": "The drug neither therapeutically changes the disease nor treats its symptoms.",
# }


# ---------------------------------------------------------------------
# Server-friendly paths
# ---------------------------------------------------------------------
# Default for Vast.ai / server:
#   /workspace/semdrug/data/paths/*.json
# You can override without editing code:
#   export SEMDRUG_ROOT=/workspace/semdrug
#   export SEMDRUG_PATHS_ROOT=/workspace/semdrug/data/paths
PROJECT_ROOT = os.getenv("SEMDRUG_ROOT", "/workspace/SemDrug")
PATHS_ROOT = os.getenv(
    "SEMDRUG_PATHS_ROOT",
    os.path.join(PROJECT_ROOT, "data", "paths"),
)

KG_DATASET_PATH = {
    "drugbank": {
        "train": os.path.join(PATHS_ROOT, "drugbank_train_add_reverse.json"),
        "test": os.path.join(PATHS_ROOT, "drugbank_test_add_reverse.json"),
    },
    "ddinter": {
        "train": os.path.join(PATHS_ROOT, "ddinter_train_add_reverse.json"),
        "test": os.path.join(PATHS_ROOT, "ddinter_test_add_reverse.json"),
    },
    "pharmacotherapydb": {
        "train": os.path.join(PATHS_ROOT, "pharmaDB_train_add_reverse.json"),
        "test": os.path.join(PATHS_ROOT, "pharmaDB_test_add_reverse.json"),
    },
}


def assert_file_exists(path: str, description: str = "file") -> str:
    if not path:
        raise ValueError(f"Missing path for {description}.")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Cannot find {description}: {path}\n"
            f"Current PROJECT_ROOT={PROJECT_ROOT}\n"
            f"Current PATHS_ROOT={PATHS_ROOT}\n"
            "Either move your data to the expected folder, set SEMDRUG_PATHS_ROOT, "
            "or pass --dataset_path explicitly."
        )
    return path

def replace_drugs_in_ddi(ddi_str):
    return ddi_str.replace("{u}", "#Drug1").replace("{v}", "#Drug2")

def load_dataset(
    dataset_path: str,
    start_index: Union[None, int] = 0,
    end_index: Union[None, int] = None,
    debug: bool = False,
) -> Dataset:
    dataset_path = assert_file_exists(dataset_path, "dataset json")
    dataset = Dataset.from_json(dataset_path)

    if start_index is not None and end_index is not None:
        end_index = min(end_index, len(dataset))
        dataset = dataset.select(range(start_index, end_index))

    if debug:
        dataset = dataset.select(range(min(250, len(dataset))))

    return dataset



def drugbank_chat_template(dataset, tokenizer, args):
    if args.few_shot:
        # load the dataset
        train_dataset = load_dataset(assert_file_exists(KG_DATASET_PATH[args.dataset_name]["train"], "few-shot train dataset"))
        # shuffle dataset
        train_dataset = train_dataset.shuffle(seed=args.seed)
        # dictionary with key as the label_idx and the value as the list of examples
        label2examples = {}
        from tqdm import tqdm

        logger.info("Creating few shot dataset")
        for example in tqdm(train_dataset.to_list()):
            label_idx = example["label_idx"]
            if label_idx not in label2examples:
                label2examples[label_idx] = []
            label2examples[label_idx].append(example)

        few_shot_data = []
        for label_idx, examples in label2examples.items():
            # take the first N = args.num_shots examples (or fewer if not enough available)
            few_shot_data.extend(examples[: args.num_shots])

        few_shot_dataset = Dataset.from_list(few_shot_data)
        # shuffle
        few_shot_dataset = few_shot_dataset.shuffle(seed=args.seed)

        # convert to str
        def process_few_shot(example):
            user_str = (
                f"Determine the interaction type between "
                f"{example['drug1_name']} (Drug1) and {example['drug2_name']} (Drug2).\n"
            )
            if args.use_drug_descriptions:
                user_str += (
                    f"Description of the drugs:\n"
                    f"{example['drug1_name']}: {example['drug1_desc']}\n"
                    f"{example['drug2_name']}: {example['drug2_desc']}\n"
                )
            if args.use_kg:
                user_str += f"Knowledge Graph Information:\n{example['path_str']}\n"

            label = example["label"]
            label = f'##Answer: {example["label"]}'
            return {"user": user_str, "assistant": label}

        few_shot_dataset = few_shot_dataset.map(
            process_few_shot, num_proc=1, remove_columns=dataset.column_names
        )
        few_shot_conv = []
        for example in few_shot_dataset.to_list():
            few_shot_conv.append({"role": "user", "content": example["user"]})
            few_shot_conv.append({"role": "assistant", "content": example["assistant"]})
    else:
        few_shot_conv = []

    def apply_template(example):
        if args.use_options:
            question = f"Question: Which of the following describes the interaction type between {example['drug1_name']} (Drug1) and {example['drug2_name']} (Drug2)?"
        else:
            question = f"Determine the interaction type between {example['drug1_name']} (Drug1) and {example['drug2_name']} (Drug2)."

        if args.use_options and args.option_style == "numbered":
            system_prompt = SYSTEM_INSTRUCTION_PROMPT["drugbank_with_options"]
        elif args.use_options and args.option_style == "bulleted":
            system_prompt = SYSTEM_INSTRUCTION_PROMPT["drugbank_with_bullets"]
        else:
            system_prompt = SYSTEM_INSTRUCTION_PROMPT["drugbank_open"]

        # add options to system prompt
        if args.use_options:
            # option_str = "\n".join(
            #     [f"- {option}" for key, option in DRUGBANK_OPTIONS.items()]
            # )

            option_str = "\n".join(
                [
                    f"- {option.replace('{u}', example['drug1_name']).replace('{v}', example['drug2_name'])}"
                    for _, option in DRUGBANK_OPTIONS.items()
                ]
            )
            system_prompt += "OPTIONS:\n"
            system_prompt += f"{option_str}\n"

        user_content = [{"role": "system", "content": system_prompt}]

        if args.few_shot:
            user_content += few_shot_conv

        test_prompt = question + "\n"

        if args.use_drug_descriptions:
            test_prompt += (
                f"Description of the drugs:\n"
                f"{example['drug1_name']}: {example['drug1_desc']}\n"  # Drug 1 name and description
                f"{example['drug2_name']}: {example['drug2_desc']}\n"  # Drug 2 name and description
            )

        if args.use_kg:
            test_prompt += (
                "Knowledge Graph Information:\n" + (example["path_str"]) + "\n"
            )

    #change add generation prompt to To True for some models
        processed_input = tokenizer.apply_chat_template(
            user_content + [{"role": "user", "content": test_prompt}],
            tokenize=False, add_generation_prompt=True
        )

        return {"input": processed_input}

    dataset = dataset.map(
        apply_template, num_proc=args.num_proc, remove_columns=dataset.column_names
    )
    return dataset

def ddinter_chat_template(dataset, tokenizer, args):
    def apply_template(example):
        question = f"Determine the severity of interaction when {example['drug1_name']} (Drug 1) and {example['drug2_name']} (Drug 2) are taken together."
        example["question"] = question

        system_prompt = SYSTEM_INSTRUCTION_PROMPT["ddinter_common"]
        test_prompt = "Question: " + example["question"] + "\n"

        user_content = system_prompt + test_prompt

        if args.use_drug_descriptions:
            user_content += (
                f"Description of the drugs:\n"
                f"{example['drug1_name']}: {example['drug1_desc']}\n"
                f"{example['drug2_name']}: {example['drug2_desc']}\n"
            )

        if args.use_kg:
            user_content += (
                "Knowledge Graph Information:\n" + example["path_str"] + "\n"
            )

        if args.use_options:
            option_str = "\n".join(
                [f"{key}: {option}" for key, option in DDINTER_OPTIONS.items()]
            )
            user_content += "Options:\n" + option_str + "\n"

        #change add generation prompt to To True for some models
        processed_input = tokenizer.apply_chat_template(
            [{"role": "user", "content": user_content}],
            tokenize=False, add_generation_prompt=True
        )

        return {"input": processed_input}

    dataset = dataset.map(
        apply_template, num_proc=args.num_proc, remove_columns=dataset.column_names
    )
    return dataset


def pharmacotherapydb_chat_template(dataset, tokenizer, args):
    def apply_template(example):
        question = (
            f"What is the therapeutic indication of {example['drug_name']} "
            f"for {example['disease_name']}?"
        )

        system_prompt = SYSTEM_INSTRUCTION_PROMPT["pharmacotherapydb"](args.use_kg)

        user_prompt = system_prompt + "\nQuestion: " + question + "\n"

        if args.use_drug_descriptions:
            user_prompt += (
                    f"Description of the drug and disease:\n"
                    f"{example['drug_name']}: {example['drug_desc']}\n"
                    f"{example['disease_name']}: {example['disease_desc']}\n"
                )

        if args.use_kg:
            user_prompt += "Knowledge Graph Information:\n" + example["path_str"] + "\n"

        if args.use_options:
            option_str = "\n".join(
                [f"- {opt}: {desc}" for opt, desc in PHARMACOTHERAPYDB_OPTIONS.items()]
            )
            user_prompt += "Options:\n" + option_str + "\n"

        messages = [
            {"role": "user", "content": user_prompt}
        ]
            
    #change add generation prompt to To True for some models
        processed_input = tokenizer.apply_chat_template(
            messages,
            tokenize=False, add_generation_prompt=True
        )
        return {"input": processed_input}

    dataset = dataset.map(
        apply_template, num_proc=args.num_proc, remove_columns=dataset.column_names
    )
    return dataset

def apply_chat_template(dataset, tokenizer, args):
    if args.dataset_name == "drugbank":
        return drugbank_chat_template(dataset, tokenizer, args)
    elif args.dataset_name == "ddinter":
        return ddinter_chat_template(dataset, tokenizer, args)
    elif args.dataset_name == "pharmacotherapydb":
        return pharmacotherapydb_chat_template(dataset, tokenizer, args)
    else:
        raise ValueError(f"Dataset {args.dataset_name} not supported")

def create_prediction_dir(args):
    # use args such as use_drug_descriptions, use_kg, use_options, option_style to create the name
    prediction_dir = ""
    # modified_model_name = args.model_name_or_path.replace("/", "_")
    prediction_dir += f"{args.dataset_path.split('/')[-1].replace('.json', '')}"
    # prediction_dir += f"_{modified_model_name}"
    # if args.use_drug_descriptions:
    #     prediction_dir += "_desc"

    if args.use_kg:
        prediction_dir += "_kg"

    # if args.few_shot:
    #     prediction_dir += f"_few_shot_{args.num_shots}"

    # if args.use_options:
    #     prediction_dir += "_options"

    #     if args.option_style == "bulleted":
    #         prediction_dir += "_bullets"

    #     if args.option_style == "numbered":
    #         prediction_dir += "_numbered"

    if args.debug:
        prediction_dir += "_debug"

    # prediction_dir += f"_seed_{args.seed}"

    return prediction_dir


def main(args):
    set_seed(args.seed)
    maybe_login_huggingface()

    # load model
    logger.info(f"Loading model {args.model_name_or_path}")
    import torch

    num_gpus = torch.cuda.device_count()
    logger.info(f"Number of GPUs: {num_gpus}")

    if num_gpus < 1:
        raise RuntimeError("No CUDA GPU detected. vLLM inference requires a GPU.")

    tensor_parallel_size = args.tensor_parallel_size or num_gpus
    tensor_parallel_size = max(1, int(tensor_parallel_size))

    llm_kwargs = {
        "model": args.model_name_or_path,
        "tensor_parallel_size": tensor_parallel_size,
        "max_model_len": args.max_model_len,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "dtype": args.dtype,
        "trust_remote_code": args.trust_remote_code,
        "max_num_seqs": args.max_num_seqs,
    }

    if args.enforce_eager:
        llm_kwargs["enforce_eager"] = True
    if args.enable_prefix_caching:
        llm_kwargs["enable_prefix_caching"] = True
    if args.download_dir:
        llm_kwargs["download_dir"] = args.download_dir

    model = LLM(**llm_kwargs)

    # load dataset
    if args.dataset_path is None:
        args.dataset_path = KG_DATASET_PATH[args.dataset_name]["test"]

    args.dataset_path = assert_file_exists(args.dataset_path, "test dataset")
    logger.info(f"Loading dataset from {args.dataset_path}")
    dataset = load_dataset(
        args.dataset_path, args.start_index, args.end_index, args.debug
    )

    logger.info("converting dataset to test examples")
    tokenizer = model.get_tokenizer()
    dataset = apply_chat_template(dataset, tokenizer, args)

    logger.info("example of dataset")
    if len(dataset) > 0:
        logger.info(dataset[0]["input"])
    else:
        raise ValueError("Dataset is empty after slicing/debug filtering.")

    # prepare prompts
    prompts = dataset["input"]

    # run inference with greedy decoding
    logger.info("Running inference")

    eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
    stop_token_ids = [tokenizer.eos_token_id]

    if isinstance(eot_id, int) and eot_id >= 0:
        stop_token_ids.append(eot_id)

    stop_strings = None

    if args.dataset_name == "ddinter":
        stop_strings = [
            "\n",
            "Question:",
            " question",
            "Based on",
            " based on",
        ]
    sampling_params = SamplingParams(
        temperature=args.temperature, top_p=args.top_p, max_tokens=args.max_tokens,
        stop=stop_strings,
        stop_token_ids=stop_token_ids,
    )
    model_outputs = model.generate(prompts, sampling_params)

    # save outputs
    predictions = [output.outputs[0].text for output in model_outputs]

    logger.info(f"Saving predictions to {args.output_dir}")
    predictions_df = pd.DataFrame({"prediction": predictions})

    prediction_dir = create_prediction_dir(args)
    output_dir = os.path.join(args.output_dir, prediction_dir)

    os.makedirs(output_dir, exist_ok=True)

    if args.start_index is not None and args.end_index is not None:
        output_path = os.path.join(
            output_dir, f"predictions_{args.start_index}_{args.end_index}.csv"
        )
    else:
        output_path = os.path.join(output_dir, "predictions.csv")

    logger.info(f"Saving predictions to {output_path}")
    predictions_df.to_csv(output_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        choices=["pharmacotherapydb", "drugbank", "ddinter"],
        help="Name of the dataset",
    )

    parser.add_argument("--max_num_seqs", type=int, default=32)

    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        help="Path to the dataset json file in case the file changes.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="meta-llama/Meta-Llama-3.1-8B-Instruct",
        help="Model name on HuggingFace or path",
    )
    parser.add_argument(
        "--download_dir",
        type=str,
        default=os.getenv("HF_HOME", None),
        help="Optional Hugging Face/vLLM download/cache directory.",
    )
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=None,
        help="Number of GPUs for tensor parallelism. Default: all visible GPUs.",
    )
    parser.add_argument("--max_model_len", type=int, default=4096)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.80)
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["auto", "half", "float16", "bfloat16", "float", "float32"],
        help="A100 supports bfloat16; use half/float16 if needed.",
    )
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--enforce_eager", action="store_true")
    parser.add_argument("--enable_prefix_caching", action="store_true")
    parser.add_argument("--output_dir", default="/workspace/SemDrug/outputs")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--num_proc", type=int, default=1)

    # smapling params
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1)
    parser.add_argument("--max_tokens", type=int, default=32)

    # experiment parameters
    parser.add_argument("--few_shot", action="store_true") 
    parser.add_argument(
        "--num_shots", type=int, default=1, help="number of shots per example"
    )  #
    parser.add_argument("--use_drug_descriptions", action="store_true")
    parser.add_argument("--use_kg", action="store_true")
    parser.add_argument("--use_options", action="store_true")
    parser.add_argument(
        "--option_style", type=str, default="bulleted", choices=["numbered", "bulleted"]
    )
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--end_index", type=int, default=None)

    args = parser.parse_args()

    main(args)