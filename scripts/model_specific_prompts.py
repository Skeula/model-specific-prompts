import modules.scripts as scripts
import gradio as gr
import csv
import os
from collections import defaultdict

import modules.shared as shared
import difflib
import random

scripts_dir = scripts.basedir()
model_mappings = None
model_mappings_modified = None

MODEL_HASH=0
MODEL_CKPT=1
PROMPT=2
NEGATIVE_PROMPT=3
IDX=4

ROW = [MODEL_HASH, MODEL_CKPT, PROMPT, NEGATIVE_PROMPT]

DEFAULT_IDX = 0
CUSTOM_IDX = 1

DEFAULT_MAPPINGS = 'default-mappings.csv'
CUSTOM_MAPPINGS = 'custom-mappings.csv'


def normalize_entry (entry, idx=None):
    row = []
    for field in ROW:
        value = entry[field].strip(' ') if field<len(entry) else ''
        row.append(value)
    if idx != None:
        row.append(idx)
    return row

def str_simularity(a, b):
    return difflib.SequenceMatcher(None, a, b).ratio()

model_hash_dict = {}

def get_old_model_hash(filename):
    if filename in model_hash_dict:
        return model_hash_dict[filename]
    try:
        with open(filename, "rb") as file:
            import hashlib
            m = hashlib.sha256()

            file.seek(0x100000)
            m.update(file.read(0x10000))
            hash = m.hexdigest()[0:8]
            model_hash_dict[filename] = hash
            return hash
    except FileNotFoundError:
        return 'NOFILE'

def get_current_model ():
    model_hash = get_old_model_hash(shared.sd_model.sd_checkpoint_info.filename)
    model_ckpt = os.path.basename(shared.sd_model.sd_checkpoint_info.filename)
    return model_hash, model_ckpt

def load_model_mappings():
    global model_mappings, model_mappings_modified, scripts_dir

    default_file = f'{scripts_dir}/{DEFAULT_MAPPINGS}'
    user_file = f'{scripts_dir}/{CUSTOM_MAPPINGS}'

    if not os.path.exists(user_file):
        open(user_file, 'w').write('\n')

    modified = str(os.stat(default_file).st_mtime) + '_' + str(os.stat(user_file).st_mtime)

    if model_mappings is None or model_mappings_modified != modified:
        model_mappings = defaultdict(list)
        def parse_file(path, idx):
            if os.path.exists(path):
                with open(path, newline='') as csvfile:
                    csvreader = csv.reader(csvfile, skipinitialspace=True)
                    for row in csvreader:
                        try:
                            if row[0].startswith('#'):
                                continue
                            ckptname = 'default' if len(row)<=2 else row[MODEL_CKPT]
                            dictrow = normalize_entry(row, idx)
                            model_mappings[row[MODEL_HASH]].append(dictrow)
                        except:
                            pass

        parse_file(default_file, DEFAULT_IDX) # 0 for default_file
        parse_file(user_file, CUSTOM_IDX) # 1 for user_file

        model_mappings_modified = modified

    return model_mappings

def get_entry_for_current_model():
    model_hash, model_ckpt = get_current_model()

    found = None

    model_mappings = load_model_mappings()

    if model_hash in model_mappings:
        lst = model_mappings[model_hash]
        found = lst[0]

        if len(lst) > 1:
            max_sim = 0.0
            for entry in lst:
                sim = str_simularity(entry[MODEL_CKPT], model_ckpt)
                if sim >= max_sim:
                    max_sim = sim
                    found = entry

    if found and found[MODEL_CKPT] == '':
        found[MODEL_CKPT] = model_ckpt

    return found

class Script(scripts.Script):
    def title(self):
        return "Model Specific Prompts"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        def check_prompt():
            entry = get_entry_for_current_model()

            if entry:
                src = f'{CUSTOM_MAPPINGS}' if entry[IDX]==1 else f'{DEFAULT_MAPPINGS} (default database)'
                return f"filename={entry[MODEL_CKPT]}\nhash={entry[MODEL_HASH]}\nprompt={entry[PROMPT]}\nnegative prompt={entry[NEGATIVE_PROMPT]}\nmatch from {src}"
            else:
                model_hash, model_ckpt = get_current_model()
                return f"filename={model_ckpt}\nhash={model_hash}\nno match"

        def edit_custom_mapping(prompt='', negative_prompt=''):
            model_hash, model_ckpt = get_current_model()

            user_file = f'{scripts_dir}/{CUSTOM_MAPPINGS}'
            tmp_user_file = f'{scripts_dir}/{CUSTOM_MAPPINGS}.tmp'
            user_backup_file = f'{scripts_dir}/{CUSTOM_MAPPINGS}.backup'
            modified = None
            with open(tmp_user_file, 'w') as outfile:
                writer = csv.writer(outfile)
                with open(user_file, newline='') as infile:
                    csvreader = csv.reader(infile, skipinitialspace=True)
                    for row in csvreader:
                        if len(row) == 0:
                            continue
                        try:
                            if row[0].startswith('#'):
                                writer.writerow(row)
                                continue
                            #row = normalize_entry(row)
                            ckptname = None if len(row)<=MODEL_CKPT else row[MODEL_CKPT]
                            if row[MODEL_HASH]==model_hash and ckptname==model_ckpt:
                                modified = normalize_entry(row)
                                continue
                            writer.writerow(row)
                        except:
                            pass
                if prompt != '' or negative_prompt != '':
                    if modified:
                        # modified is the row we're replacing, so we pull in
                        # default values from it
                        if prompt == '':
                            prompt = modified[PROMPT]
                        if negative_prompt == '':
                            negative_prompt = modified[NEGATIVE_PROMPT]
                    
                    modified = normalize_entry([model_hash, model_ckpt, prompt, negative_prompt])
                    writer.writerow(modified)
            if modified:
                try:
                    os.unlink(user_backup_file)
                except:
                    pass
                try:
                    os.rename(user_file, user_backup_file)
                except:
                    os.unlink(user_file)
                    pass
                os.rename(tmp_user_file, user_file)
            else:
                unlink(tmp_user_file)

            return modified

        def delete_prompt():
            found = edit_custom_mapping()
            if found:
                return f'deleted entry: {found}'
            else:
                return f'no custom mapping found'

        def add_custom(prompt, negative_prompt):
            if len(prompt)+len(negative_prompt) == 0:
                return "Fill model specific prompts"
            modified = edit_custom_mapping(prompt, negative_prompt)

            return f'added: hash={modified[MODEL_HASH]}, ckpt={modified[MODEL_CKPT]}, prompt={modified[PROMPT]}, negative_prompt={modified[NEGATIVE_PROMPT]} -- {modified}'

        with gr.Group():
            with gr.Accordion('Model Specific Prompts', open=False):
                is_enabled = gr.Checkbox(label='Model Specific Prompts Enabled', value=True)

                prompt_placement = gr.Dropdown(choices=["model-prompt, your-prompt", "your-prompt, model-prompt", "model-prompt your-prompt", "your-prompt model-prompt"], 
                                value='model-prompt, your-prompt',
                                label='Model prompt placement:')

                with gr.Accordion('Add Custom Mappings', open=False):
                    info = gr.HTML(f"<p style=\"margin-bottom:0.75em\">Set prompts to add to the current model. Custom mappings are saved to extensions/model-specific-prompts/{CUSTOM_MAPPINGS}</p>")
                    prompt_input = gr.Textbox(placeholder="Prompt text", label="Prompt")
                    negative_input = gr.Textbox(placeholder="Prompt text", label="Negative Prompt")
                    with gr.Row():
                        check_mappings = gr.Button(value='Check')
                        add_mappings = gr.Button(value='Save')
                        delete_mappings = gr.Button(value='Delete')

                    text_output = gr.Textbox(interactive=False, label='result')

                    add_mappings.click(add_custom, inputs=[prompt_input,negative_input], outputs=text_output)
                    check_mappings.click(check_prompt, inputs=None, outputs=text_output)
                    delete_mappings.click(delete_prompt, inputs=None, outputs=text_output)


        return [is_enabled, prompt_placement]

    def process(self, p, is_enabled, prompt_placement):

        if not is_enabled:
            global model_mappings
            model_mappings = None
            return

        def new_prompt(prompt, model_prompt):
            arr = [model_prompt]

            if prompt_placement.startswith('model'):
                arr.append(prompt)
            else:
                arr.insert(0, prompt)

            return ', '.join(arr)

        entry = get_entry_for_current_model()

        if entry:
            if entry[PROMPT] != '':
                p.prompt = new_prompt(p.prompt, entry[PROMPT])
                p.all_prompts = [new_prompt(prompt, entry[PROMPT]) for prompt in p.all_prompts]

            if entry[NEGATIVE_PROMPT] != '':
                p.negative_prompt = new_prompt(p.negative_prompt, entry[NEGATIVE_PROMPT])
                p.all_negative_prompts = [new_prompt(negative_prompt, entry[NEGATIVE_PROMPT]) for negative_prompt in p.all_negative_prompts]
